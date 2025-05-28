import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
import pickle
from data_utils import *
from dataset import get_dataloader, load_dataset
from model import CARCA, SENS_MAPPING
from evaluate import Evaluator
import os


def bce_loss(pos_logits, neg_logits, mask=None):
    # mask: (batch, maxlen), float tensor (1 for valid, 0 for pad)
    # If mask is not provided, assume all are valid (e.g. for single item pred)
    if mask is None:
        mask = torch.ones_like(pos_logits)

    loss = (
        -torch.log(torch.sigmoid(pos_logits) + 1e-24) * mask
        - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * mask
    )
    # Ensure sum over valid entries only before dividing
    # Handle cases where mask.sum() could be zero to avoid NaN
    sum_loss = loss.sum()
    sum_mask = mask.sum()
    if sum_mask > 0:
        return sum_loss / sum_mask
    return torch.tensor(0.0, device=pos_logits.device) # Or handle as appropriate


def train_one_epoch(model, dataloader, optimizer_main, optimizer_mine, device, fairness_lambda):
    model.train()
    total_loss = 0.0
    total_rec_loss = 0.0
    total_fairness_loss = 0.0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        
        optimizer_main.zero_grad()
        optimizer_mine.zero_grad()

        user_feat_all = batch["user_feat"]
        
        # Model forward pass now returns mi_estimate
        pos_logits, neg_logits, mi_est = model(
            user_feat_all, # This is user_feat_all for CARCA model
            # user_feat_all, # Second arg was sens_feat_all in reference, CARCA handles it internally from user_feat_all
            batch["seq"],
            batch["seq_feat"],
            batch["seqcxt"],
            batch["pos"],
            batch["pos_feat"],
            batch["poscxt"],
            batch["neg"],
            batch["neg_feat"],
            batch["negcxt"],
        )
        
        mask = (batch["pos"] != 0).float() # Assuming positive items define valid interactions for loss
        rec_loss = bce_loss(pos_logits, neg_logits, mask)
        fairness_loss = mi_est.mean() # Average MI over batch
        
        loss = rec_loss + fairness_lambda * fairness_loss
        
        loss.backward()
        optimizer_main.step()
        optimizer_mine.step()
        
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_fairness_loss += fairness_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_rec_loss = total_rec_loss / len(dataloader)
    avg_fairness_loss = total_fairness_loss / len(dataloader)
    return avg_loss, avg_rec_loss, avg_fairness_loss


def load_split(split_name, out_dir):
    with open(os.path.join(out_dir, split_name), "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--maxlen", type=int, default=75)
    parser.add_argument("--hidden_units", type=int, default=90)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--l2_emb", type=float, default=0.0001)
    parser.add_argument("--cxt_size", type=int, default=6)
    parser.add_argument("--use_res", type=bool, default=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--selected_sens", type=str, default="0", 
                        help="Comma-separated indices or names of sensitive attributes (e.g., '0,1' or 'gender,age')")
    parser.add_argument("--fairness_lambda", type=float, default=0.1,
                        help="Weight for mutual information fairness loss")
    parser.add_argument("--mine_lr", type=float, default=1e-5, # Default MINE LR from reference
                        help="Learning rate for MINE optimizer")
    args = parser.parse_args()

    (
        user_train,
        user_features,
        itemnum,
        cxtdict,
        cxtsize,
        maxlen,
        item_features,
        usernum,
        itemid2idx,
    ) = load_dataset(args.dataset, maxlen=args.maxlen, cxt_size=args.cxt_size)

    dataloader = get_dataloader(
        user_train,
        user_features,
        itemnum,
        cxtdict,
        cxtsize,
        maxlen,
        args.batch_size,
        item_features,
        itemid2idx=itemid2idx,
    )

    # Process selected_sens
    # If attribute names like 'gender', 'age' are used, they are mapped by SENS_MAPPING in model
    # Here we just split the string for the model constructor
    selected_sens_list = [s.strip() for s in args.selected_sens.split(',')]

    model = CARCA(
        usernum,
        itemnum,
        args,
        item_features.shape[1],
        user_features.shape[1],
        cxtsize,
        selected_sens=selected_sens_list, # Pass processed list
        sens_feature_dim=len(selected_sens_list) # Pass number of selected attributes
    ).to(args.device)

    # New optimizers
    main_params = [p for n, p in model.named_parameters() if "mine." not in n and p.requires_grad]
    mine_params = [p for n, p in model.named_parameters() if "mine." in n and p.requires_grad]

    optimizer_main = optim.Adam(main_params, lr=args.lr)
    optimizer_mine = optim.Adam(mine_params, lr=args.mine_lr)

    # Load validation split for ml-1m
    out_dir = "Data/movielens_preprocessed"
    user_train_split = load_split("user_train_split.pkl", out_dir)
    user_valid = load_split("user_valid.pkl", out_dir)

    evaluator = Evaluator(
        model,
        user_features,
        item_features,
        cxtdict,
        itemid2idx,
        args.device,
    )

    # Set up model save directory
    save_dir = os.path.join("saved_models", args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_ndcg20 = -1
    for epoch in range(1, args.num_epochs + 1):
        avg_loss, avg_rec_loss, avg_fairness_loss = train_one_epoch(
            model, dataloader, optimizer_main, optimizer_mine, args.device, args.fairness_lambda
        )
        print(f"Epoch {epoch:02d} | Total Loss: {avg_loss:.4f} | Rec Loss: {avg_rec_loss:.4f} | Fair Loss: {avg_fairness_loss:.4f}")
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            user_valid_subset = evaluator.sample_user_subset(user_valid, percent=0.1)
            metrics = evaluator.evaluate(user_train_split, user_valid_subset, k=20, batch_size=32, candidate_chunk_size=200, fairness_metrics=False)
            print("Validation metrics (10% subset):")
            for k in [1, 5, 10, 20]:
                print(f"  NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}  Hit@{k}: {metrics[f'hit@{k}']:.4f}  MRR@{k}: {metrics[f'mrr@{k}']:.4f}")
            print(f"  DP_gender: {metrics['dp_gender']:.4f}")
            print(f"  DP_age: {metrics['dp_age']:.4f}")
            ndcg20 = metrics['ndcg@20']
            if ndcg20 > best_ndcg20:
                best_ndcg20 = ndcg20
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch} with NDCG@20: {ndcg20:.4f}")
    print(f"Best Validation NDCG@20: {best_ndcg20:.4f}")

    # After training, evaluate on the test set
    print("\nEvaluating on the test set with the best model...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    # Use the full test set
    if hasattr(evaluator, 'user_test') and evaluator.user_test is not None:
        user_test_set = evaluator.user_test
    else:
        # Try to load from file if not present
        out_dir = "Data/movielens_preprocessed"
        try:
            with open(os.path.join(out_dir, "user_test.pkl"), "rb") as f:
                user_test_set = pickle.load(f)
        except Exception as e:
            print("Test set not found. Skipping test evaluation.")
            user_test_set = None
    if user_test_set is not None:
        test_metrics = evaluator.evaluate(user_train_split, user_test_set, k=20, batch_size=32, candidate_chunk_size=200)
        print("Test set metrics:")
        # Print as a table
        print("  |   k   | NDCG  | Hit   |  MRR  |")
        print("  |-------|-------|-------|-------|")
        for k in [1, 5, 10, 20]:
            print(f"  | {k:<5} | {test_metrics[f'ndcg@{k}']:.4f} | {test_metrics[f'hit@{k}']:.4f} | {test_metrics[f'mrr@{k}']:.4f} |")
        print("  |-------|-------|-------|-------|")
        print(f"  DP_gender: {test_metrics['dp_gender']:.4f}")
        print(f"  DP_age: {test_metrics['dp_age']:.4f}")
    else:
        print("No test set available for evaluation.")


if __name__ == "__main__":
    main()

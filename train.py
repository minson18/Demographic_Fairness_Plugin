import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
import pickle
from data_utils import *
from dataset import get_dataloader, load_dataset
from model import CARCA
from evaluate import Evaluator
import os
import json


def bce_loss(pos_logits, neg_logits, mask):
    # mask: (batch, maxlen), float tensor (1 for valid, 0 for pad)
    loss = (
        -torch.log(torch.sigmoid(pos_logits) + 1e-24) * mask
        - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * mask
    )
    return loss.sum() / mask.sum()


def train_one_epoch(model, dataloader, optimizer, device, fairness_lambda=0.0):
    model.train()
    total_loss = 0
    task_loss_sum = 0
    fairness_loss_sum = 0
    
    for batch in tqdm(dataloader, desc="Train", leave=False):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        optimizer.zero_grad()
        pos_logits, neg_logits, user_repr = model(
            batch["user_feat"],
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
        mask = (batch["seq"] != 0).float()
        task_loss = bce_loss(pos_logits, neg_logits, mask)
        
        # Add fairness regularization if enabled
        fairness_loss = 0.0
        if fairness_lambda > 0:
            fairness_loss = model.compute_fairness_loss(user_repr, batch["user_feat"])
            
        # Combined loss
        loss = task_loss + fairness_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        task_loss_sum += task_loss.item()
        fairness_loss_sum += fairness_loss.item() if isinstance(fairness_loss, torch.Tensor) else fairness_loss
        
    avg_loss = total_loss / len(dataloader)
    avg_task_loss = task_loss_sum / len(dataloader)
    avg_fairness_loss = fairness_loss_sum / len(dataloader)
    
    return avg_loss, avg_task_loss, avg_fairness_loss


def load_split(split_name, out_dir):
    with open(os.path.join(out_dir, split_name), "rb") as f:
        return pickle.load(f)


def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif hasattr(obj, "item") and callable(obj.item):
        # Handles numpy scalars
        return obj.item()
    else:
        return obj


def train():
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
    parser.add_argument("--save_dir", type=str, default=None)
    
    # CUFRL fairness parameters
    parser.add_argument("--use_fairness", action="store_true", help="Enable universal fairness")
    parser.add_argument("--fairness_lambda", type=float, default=0.0, 
                       help="Weight for fairness loss (0 to disable, >0 to enable)")
    parser.add_argument("--sensitive_indices", nargs="+", type=int, default=[0, 1],
                       help="Indices of sensitive attributes in user features")
    
    args, _ = parser.parse_known_args()
    # Log training parameters
    print("Training parameters:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max sequence length: {args.maxlen}")
    print(f"  Hidden units: {args.hidden_units}")
    print(f"  Number of blocks: {args.num_blocks}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  L2 regularization: {args.l2_emb}")
    print(f"  Context size: {args.cxt_size}")
    print(f"  Use residual: {args.use_res}")
    print(f"  Device: {args.device}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Fairness enabled: {args.use_fairness}")
    print(f"  Fairness lambda: {args.fairness_lambda}")
    print(f"  Sensitive attribute indices: {args.sensitive_indices}")
    print()
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
    model = CARCA(
        usernum,
        itemnum,
        args,
        item_features.shape[1],
        user_features.shape[1],
        cxtsize,
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
    user_valid_subset = evaluator.sample_user_subset(user_valid, percent=0.3)

    # Set up model save directory
    save_dir = (
        args.save_dir
        if args.save_dir is not None
        else os.path.join("saved_models", args.dataset)
    )
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    best_ndcg20 = -1
    for epoch in range(1, args.num_epochs + 1):
        loss, task_loss, fairness_loss = train_one_epoch(
            model, dataloader, optimizer, args.device, args.fairness_lambda
        )
        print(f"Epoch {epoch}, Total Loss: {loss:.4f}, Task Loss: {task_loss:.4f}, Fairness Loss: {fairness_loss:.4f}")
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            metrics = evaluator.evaluate(
                user_train_split,
                user_valid_subset,
                k=20,
                batch_size=32,
                candidate_chunk_size=200,
                fairness_metrics=True,  # Always evaluate fairness metrics
            )
            print("Validation metrics (30% subset):")
            Evaluator.print_metrics_table(metrics)
            ndcg20 = metrics["ndcg@20"]
            
            # Optionally incorporate fairness into model selection criteria
            if args.use_fairness and args.fairness_lambda > 0:
                # Use a combined metric that considers both accuracy and fairness
                fairness_score = 0
                if "distance_gender" in metrics:
                    fairness_score += metrics["distance_gender"]
                if "distance_age" in metrics:
                    fairness_score += metrics["distance_age"]
                if "distance_occupation" in metrics:
                    fairness_score += metrics["distance_occupation"]
                
                # Normalize fairness score (lower is better)
                fairness_score = fairness_score / 3 if fairness_score > 0 else 0
                
                # Combined score: maximize NDCG, minimize fairness disparity
                combined_score = ndcg20 - args.fairness_lambda * fairness_score
                
                if combined_score > best_ndcg20:
                    best_ndcg20 = combined_score
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved at epoch {epoch} with combined score: {combined_score:.4f} (NDCG@20: {ndcg20:.4f}, Fairness: {fairness_score:.4f})")
            else:
                # Traditional model selection based on NDCG only
                if ndcg20 > best_ndcg20:
                    best_ndcg20 = ndcg20
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved at epoch {epoch} with NDCG@20: {ndcg20:.4f}")
    
    if args.use_fairness:
        print(f"Best Combined Score: {best_ndcg20:.4f}")
    else:
        print(f"Best Validation NDCG@20: {best_ndcg20:.4f}")

    # Load the best model before final validation evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    model.eval()

    # After training, evaluate on the full validation set for grid search selection
    full_val_metrics = evaluator.evaluate(
        user_train_split,
        user_valid,  # full validation set
        k=20,
        batch_size=32,
        candidate_chunk_size=200,
        fairness_metrics=True,  # Always evaluate fairness for final metrics
    )
    metrics_path = os.path.join(save_dir, "val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(convert_to_native(full_val_metrics), f, indent=2)


if __name__ == "__main__":
    train()

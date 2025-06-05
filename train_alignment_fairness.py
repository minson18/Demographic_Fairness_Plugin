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


def binary_quantile_loss(pos_logits, neg_logits, mask, q=0.8, positive_weight=0.7):
    pred_pos = torch.sigmoid(pos_logits)
    pred_neg = torch.sigmoid(neg_logits)

    y_pos = torch.ones_like(pos_logits)
    y_neg = torch.zeros_like(neg_logits)

    pos_error = y_pos - pred_pos
    neg_error = y_neg - pred_neg

    pos_loss = torch.max(q * pos_error, (q - 1) * pos_error) * mask
    neg_loss = torch.max(q * neg_error, (q - 1) * neg_error) * mask

    total_loss = (
        positive_weight * pos_loss.sum() + (1 - positive_weight) * neg_loss.sum()
    )

    return total_loss / mask.sum()


def ranking_quantile_loss(pos_logits, neg_logits, mask, q=0.8):
    # pos_logits, neg_logits: (batch, maxlen)
    # mask: (batch, maxlen) → 1 = valid, 0 = pad

    margin = 1.0
    diff = pos_logits - neg_logits  # (batch, maxlen)
    error = margin - diff  # Higher when pos < neg

    quantile_loss = torch.max(q * error, (q - 1) * error)
    masked_loss = quantile_loss * mask

    return masked_loss.sum() / mask.sum()


def bce_loss(pos_logits, neg_logits, mask):
    # mask: (batch, maxlen), float tensor (1 for valid, 0 for pad)
    loss = (
        -torch.log(torch.sigmoid(pos_logits) + 1e-24) * mask
        - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * mask
    )
    return loss.sum() / mask.sum()


def mean_align_by_attribute(x_pred, user_feat_np, attr_name):
    import numpy as np # Keep numpy import if needed elsewhere, but not for mean calculation
    import torch
    # Determine column index for each attribute
    attr_map = {"gender": 0, "age": 1, "occupation": 2}
    attr_idx = attr_map[attr_name]
    group_values = user_feat_np[:, attr_idx]

    # Create a tensor to store means for each user in the batch
    # This will ensure it's on the correct device and maintains gradient flow for means if needed
    # (though typically the group means themselves are not directly differentiated w.r.t. model params)
    # The crucial part is that the mean is calculated on a tensor that *is* part of the graph.
    batch_means = torch.zeros(x_pred.shape[0], device=x_pred.device, dtype=x_pred.dtype)

    unique_groups = np.unique(group_values) # Get unique groups from the numpy array

    for g in unique_groups:
        # Create a boolean mask on the torch tensor x_pred directly
        # Ensure group_values is a torch tensor for direct comparison if preferred, or use numpy mask then convert
        group_mask_np = (group_values == g) # This is a numpy boolean array
        group_mask_torch = torch.from_numpy(group_mask_np).to(x_pred.device)

        if group_mask_torch.sum() > 0: # Only calculate mean if there are elements in the group
            # Calculate the mean using torch operations, maintaining the graph
            group_mean = x_pred[group_mask_torch].mean()
            batch_means[group_mask_torch] = group_mean # Assign the mean to all users in this group

    # Expand batch_means to match the maxlen dimension for subtraction
    aligned_means_expanded = batch_means.unsqueeze(1).expand_as(x_pred)

    aligned = x_pred - aligned_means_expanded
    return aligned

def train_one_epoch(model, dataloader, optimizer, device, alpha=0.05, beta=0.2, debias_method="none", debias_attributes=[]):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        optimizer.zero_grad()

        # Call the model to get initial logits *before* debiasing
        pos_logits, neg_logits = model(
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

        # Apply debiasing alignment if specified
        if debias_method != "none" and len(debias_attributes) > 0:
            user_feat_np = batch["user_feat"].detach().cpu().numpy()
            for attr in debias_attributes:
                if debias_method == "mean":
                    pos_logits = mean_align_by_attribute(pos_logits, user_feat_np, attr)

        mask = (batch["seq"] != 0).float()

        # Combine three different losses together with ratio
        BCEloss = bce_loss(pos_logits, neg_logits, mask)
        quantile_loss = binary_quantile_loss(
            pos_logits, neg_logits, mask, q=0.8, positive_weight=0.7
        )
        quantile_loss_rank = ranking_quantile_loss(pos_logits, neg_logits, mask, q=0.8)
        loss = (
            BCEloss * (1 - alpha - beta)
            + quantile_loss * alpha
            + quantile_loss_rank * beta
        )
       # loss = BCEloss
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


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
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--debias_method", type=str, choices=["mean", "none"], default="none")
    parser.add_argument("--debias_attributes", nargs="+", type=str, choices=["gender", "age", "occupation"], default=[])

    # For KUAISHOU
    parser.add_argument("--alpha", type=float, default=0.2)  # for quantile loss
    parser.add_argument("--beta", type=float, default=0.2)  # for ranking quantile loss
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
    print(f"  Save directory: {args.model_dir}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Beta: {args.beta}")
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


    # Load validation split for ml-quan
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
    model_dir = (
        args.model_dir
        if args.model_dir is not None
        else os.path.join("saved_models", args.dataset)
    )
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.pth")

    best_ndcg20 = -1
    for epoch in range(1, args.num_epochs + 1):

        loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            args.device,
            args.alpha,
            args.beta,
            args.debias_method,  # Add this
            args.debias_attributes  # Add this
        )
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            metrics = evaluator.evaluate(
                user_train_split,
                user_valid_subset,
                k=20,
                batch_size=32,
                candidate_chunk_size=200,
                fairness_metrics=False,
            )
            print("Validation metrics (30% subset):")
            Evaluator.print_metrics_table(metrics)
            ndcg20 = metrics["ndcg@20"]
            if ndcg20 > best_ndcg20:
                best_ndcg20 = ndcg20
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch} with NDCG@20: {ndcg20:.4f}")
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
        fairness_metrics=False,
    )
    metrics_path = os.path.join(model_dir, "val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(convert_to_native(full_val_metrics), f, indent=2)


if __name__ == "__main__":
    train()

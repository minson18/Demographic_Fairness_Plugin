import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from data_utils import *
from dataset import get_dataloader
from cufrl_model import CUFRLModel
import torch.nn.functional as F


def fairness_loss_fn(fairness_predicted_sensitive_attrs, sensitive_attrs, reduction='mean'):
    return F.binary_cross_entropy_with_logits(fairness_predicted_sensitive_attrs, sensitive_attrs.float(), reduction=reduction)


def reconstruction_loss_fn(reconstructed_sensitive_attrs, sensitive_attrs, reduction='mean'):
    return F.binary_cross_entropy_with_logits(reconstructed_sensitive_attrs, sensitive_attrs.float(), reduction=reduction)


def bce_loss(pos_logits, neg_logits):
    istarget = (pos_logits != 0).float()
    loss = (
        -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
        - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
    )
    return loss.sum() / istarget.sum()


def risk_aware_loss(pos_logits, neg_logits, risk_weights):
    istarget = (pos_logits != 0).float()
    pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget * risk_weights
    neg_loss = -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
    loss = pos_loss + neg_loss
    return loss.sum() / istarget.sum()


def train_one_epoch(model, dataloader, optimizer, device, use_risk_weights=False, fairness_lambda=0.1, reconstruction_lambda=0.1):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        
        optimizer.zero_grad()
        
        # Determine if we have financial features in the batch
        financial_features = batch.get("financial_features", None)
        sensitive_attrs = batch.get("sensitive_attrs", None)
        
        pos_logits, neg_logits, reconstructed_sensitive_attrs, fairness_predicted_sensitive_attrs = model(
            batch["user_feat"],
            sensitive_attrs,
            batch["seq"],
            batch["seq_feat"],
            batch["seqcxt"],
            batch["pos"],
            batch["pos_feat"],
            batch["poscxt"],
            batch["neg"],
            batch["neg_feat"],
            batch["negcxt"],
            financial_features,
        )
        
        if use_risk_weights and "risk_weights" in batch:
            recommendation_loss = risk_aware_loss(pos_logits, neg_logits, batch["risk_weights"])
        else:
            recommendation_loss = bce_loss(pos_logits, neg_logits)
        
        # Calculate fairness and reconstruction losses
        fairness_loss = torch.tensor(0.0).to(device)
        if fairness_predicted_sensitive_attrs is not None and sensitive_attrs is not None:
            fairness_loss = fairness_loss_fn(fairness_predicted_sensitive_attrs, sensitive_attrs)

        reconstruction_loss_val = torch.tensor(0.0).to(device)
        if reconstructed_sensitive_attrs is not None and sensitive_attrs is not None:
            reconstruction_loss_val = reconstruction_loss_fn(reconstructed_sensitive_attrs, sensitive_attrs)

        # Combine losses
        loss = recommendation_loss + fairness_lambda * fairness_loss + reconstruction_lambda * reconstruction_loss_val
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Finance")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--maxlen", type=int, default=75)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2_emb", type=float, default=0.0001)
    parser.add_argument("--cxt_size", type=int, default=10)  # Enhanced context size
    parser.add_argument("--use_risk", type=bool, default=True)  # Use risk-aware loss
    parser.add_argument("--use_financial_features", type=bool, default=True)
    parser.add_argument("--sensitive_attribute_dim", type=int, default=1)
    parser.add_argument("--fairness_lambda", type=float, default=0.1)
    parser.add_argument("--reconstruction_lambda", type=float, default=0.1)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # Data loading
    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = dataset
    
    # Load appropriate features based on dataset
    if args.dataset == "Finance":
        ItemFeatures = get_finance_item_features(itemnum)
        UserFeatures = get_finance_user_features(usernum)
        FinancialFeatures = get_financial_risk_profiles(usernum) if args.use_financial_features else None
        CXTDict = load_data("./Data/CXTDictFinance.dat")
        SensitiveAttributes = get_sensitive_attributes(usernum)
    else:
        # Fall back to existing datasets
        if args.dataset == "Beauty":
            ItemFeatures = get_ItemDataBeauty(itemnum)
            UserFeatures = get_UserDataBeauty(usernum)
            CXTDict = load_data("./Data/CXTDictSasRec_Beauty.dat")
        elif args.dataset == "Men":
            ItemFeatures = get_ItemDataMen(itemnum)
            UserFeatures = get_UserDataMen(usernum)
            CXTDict = load_data("./Data/CXTDictSasRec_Men.dat")
        elif args.dataset == "Fashion":
            ItemFeatures = get_ItemDataFashion(itemnum)
            UserFeatures = get_UserDataFashion(usernum)
            CXTDict = load_data("./Data/CXTDictSasRec_Fashion.dat")
        elif args.dataset == "Video_Games":
            ItemFeatures = get_ItemDataGames(itemnum)
            UserFeatures = get_UserDataFashion(usernum)
            CXTDict = load_data("./Data/CXTDictSasRec_Games.dat")
        else:
            raise ValueError("Unknown dataset")
        FinancialFeatures = None
        SensitiveAttributes = get_sensitive_attributes(usernum)

    # Create dataloader with financial features if available
    dataloader = get_financial_dataloader(
        user_train,
        UserFeatures,
        itemnum,
        CXTDict,
        args.cxt_size,
        args.maxlen,
        args.batch_size,
        ItemFeatures,
        FinancialFeatures,
        SensitiveAttributes,
    ) if args.dataset == "Finance" else get_dataloader(
        user_train,
        UserFeatures,
        itemnum,
        CXTDict,
        args.cxt_size,
        args.maxlen,
        args.batch_size,
        ItemFeatures,
        SensitiveAttributes,
    )

    # Initialize the CUFRL model
    financial_feature_dim = FinancialFeatures.shape[1] if FinancialFeatures is not None else 0
    sensitive_attribute_dim = SensitiveAttributes.shape[1] if SensitiveAttributes is not None else 0
    model = CUFRLModel(
        usernum,
        itemnum,
        args,
        ItemFeatures.shape[1],
        UserFeatures.shape[1],
        args.cxt_size,
        financial_feature_dim,
        sensitive_attribute_dim,
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_emb)

    for epoch in range(1, args.num_epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, args.device, args.use_risk, args.fairness_lambda, args.reconstruction_lambda)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), f"cufrl_model_{args.dataset}.pt")
    print(f"Model saved as cufrl_model_{args.dataset}.pt")


if __name__ == "__main__":
    main() 
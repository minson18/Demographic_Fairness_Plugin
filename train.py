import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from data_utils import *
from dataset import get_dataloader, load_dataset
from model import CARCA


def bce_loss(pos_logits, neg_logits):
    istarget = (pos_logits != 0).float()
    loss = (
        -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
        - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
    )
    return loss.sum() / istarget.sum()


# def train_one_epoch(model, dataloader, optimizer, device, fairness_lambda=0.1):
def train_one_epoch(model, dataloader, optimizer_main, optimizer_mine, device, fairness_lambda):
    """
    單 epoch 訓練：
      1) 正常 forward → pos_logits, neg_logits, mi_est
      2) rec_loss = BCE(pos, neg)
      3) fairness_loss = mean(mi_est)
      4) loss = rec_loss + fairness_lambda * fairness_loss
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        # 移動到 device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        # optimizer.zero_grad()
        optimizer_main.zero_grad()
        optimizer_mine.zero_grad()
        # 1) 拆出完整的 user_feat 與敏感屬性 sens_feat
        user_feat_all = batch["user_feat"]  # (batch, user_feature_dim)
        # model.sens_indices 於 model.py __init__ 時設定
        sens_feat = user_feat_all[:, model.sens_indices]  # (batch, len(selected_sens))
        # 2) Forward：取得推薦 logits + 互資訊估計
        pos_logits, neg_logits, mi_est = model(
            user_feat_all,
            # sens_feat,
            user_feat_all, # Let the model extract the sensitive features itself
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
        # 3) 計算損失：推薦任務 + fairness 正則化
        rec_loss = bce_loss(pos_logits, neg_logits)
        fairness_loss = mi_est.mean()  # 對 batch 取平均
        loss = rec_loss + fairness_lambda * fairness_loss
        # 4) 反向並更新
        loss.backward()
        # optimizer.step()
        optimizer_main.step()
        optimizer_mine.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--maxlen", type=int, default=75)
    parser.add_argument("--hidden_units", type=int, default=90)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--l2_emb", type=float, default=0.0001)
    parser.add_argument("--cxt_size", type=int, default=6)
    parser.add_argument("--use_res", type=bool, default=True)
    parser.add_argument("--selected_sens", type=str, default="0", 
                    help="Comma-separated indices of sensitive attributes")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # 新增 fairness 權重參數
    parser.add_argument("--fairness_lambda", type=float, default=0.1,
                        help="weight for mutual information fairness loss")
    parser.add_argument("--mine_lr", type=float, default=1e-5,
                        help="learning rate for MINE optimizer")
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
    print(f"User features shape: {user_features.shape}")
    print(f"First few user features: {user_features[:3]}")
    # Then when initializing the model, modify to:
    # selected_sens = [int(idx) for idx in args.selected_sens.split(",")]
    selected_sens = args.selected_sens.split(",")
    model = CARCA(
        usernum,
        itemnum,
        args,
        item_features.shape[1],
        user_features.shape[1],
        selected_sens=selected_sens,
        sens_feature_dim=len(selected_sens),
        cxt_size=cxtsize,
    ).to(args.device)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 主模型 Optimizer（不含 MINE)
    main_params = [p for n,p in model.named_parameters() if "mine." not in n]
    optimizer_main = optim.Adam(main_params, lr=args.lr)
    # MINE 的 Optimizer（專門更新 Mutual Information Estimator）
    optimizer_mine = optim.Adam(model.mine.parameters(), lr=args.mine_lr)

    for epoch in range(1, args.num_epochs + 1):
        # loss = train_one_epoch(model, dataloader, optimizer, args.device, args.fairness_lambda)
        loss = train_one_epoch(
            model,
            dataloader,
            optimizer_main,
            optimizer_mine,
            args.device,
            args.fairness_lambda
        )
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


if __name__ == "__main__":
    main()

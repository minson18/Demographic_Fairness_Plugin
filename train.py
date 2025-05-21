import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from data_utils import data_partition, load_data, get_ItemDataBeauty, get_UserDataBeauty, \
                       get_ItemDataMen, get_UserDataMen, get_ItemDataFashion, get_UserDataFashion, \
                       get_ItemDataGames, load_sensitive_attributes
from dataset import get_dataloader
from model import SequentialRecModel
import math


def bce_loss(pos_logits, neg_logits, istarget_override=None):
    istarget = istarget_override if istarget_override is not None else (pos_logits != 0).float()
    loss = (
        -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
        - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
    )
    target_sum = istarget.sum()
    if target_sum > 0:
        return loss.sum() / target_sum
    return torch.tensor(0.0, device=pos_logits.device)


def train_one_epoch(model, dataloader, optimizer, device, fairness_lambda):
    model.train()
    total_rec_loss = 0
    total_fairness_loss = 0
    total_combined_loss = 0
    
    for batch in tqdm(dataloader, desc="Train", leave=False):
        # 將所有張量移至設備
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        
        optimizer.zero_grad()
        
        # 前向傳播
        pos_logits, neg_logits, fairness_output = model(
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
            batch["sensitive_attribute"]
        )
        
        # 計算推薦損失（僅對非填充項）
        istarget = (batch["seq"] != 0).float()
        rec_loss = bce_loss(pos_logits, neg_logits, istarget_override=istarget)
        
        # 互信息估計作為公平性損失
        # 我們希望最小化用戶表示和敏感屬性之間的互信息
        # 使用絕對值確保互信息非負
        fairness_loss = torch.abs(fairness_output).mean()
        
        # 計算帶權重的組合損失
        # 如果fairness_lambda為負，則鼓勵互信息（不公平，但可能提高性能）
        # 如果fairness_lambda為正，則懲罰互信息（提高公平性，可能降低性能）
        combined_loss = rec_loss + fairness_lambda * fairness_loss
        
        # 反向傳播和優化
        combined_loss.backward()
        optimizer.step()
        
        # 累計損失統計
        total_rec_loss += rec_loss.item()
        total_fairness_loss += fairness_loss.item()
        total_combined_loss += combined_loss.item()
    
    num_batches = len(dataloader)
    return (
        total_rec_loss / num_batches,
        total_fairness_loss / num_batches,
        total_combined_loss / num_batches,
    )


def evaluate(model, user_valid, item_features, context_dict, user_features, device, sensitive_attributes=None, k_list=[5, 10, 20], maxlen=50):
    """
    評估模型在驗證集上的性能，計算常見的排序指標。
    
    Args:
        model: 要評估的模型
        user_valid: 驗證集，{user_id: [item_ids]}
        item_features: 項目特徵矩陣
        context_dict: 上下文特徵字典
        user_features: 用戶特徵矩陣
        device: 設備（CPU或GPU）
        sensitive_attributes: 可選，敏感屬性映射，用於計算公平性指標
        k_list: 排序指標的k值列表
        maxlen: 序列最大長度
        
    Returns:
        metrics: 包含不同指標的字典
    """
    model.eval()
    metrics = {f'NDCG@{k}': 0.0 for k in k_list}
    metrics.update({f'HR@{k}': 0.0 for k in k_list})
    
    # 按敏感屬性分類的指標（如果提供）
    if sensitive_attributes:
        sensitive_groups = {}
        for user, attr in sensitive_attributes.items():
            if attr not in sensitive_groups:
                sensitive_groups[attr] = []
            sensitive_groups[attr].append(user)
        
        for attr in sensitive_groups:
            for k in k_list:
                metrics[f'NDCG@{k}_group{attr}'] = 0.0
                metrics[f'HR@{k}_group{attr}'] = 0.0
        
        # 計算組間差異的指標
        group_counts = {attr: len(users) for attr, users in sensitive_groups.items()}
        metrics['group_counts'] = group_counts
    
    valid_users = 0
    
    with torch.no_grad():
        for user in tqdm(user_valid, desc="Evaluate"):
            if len(user_valid[user]) == 0:
                continue
                
            valid_users += 1
            target_item = user_valid[user][0]  # 驗證集中的目標項目
            
            # 獲取用戶的訓練序列和特徵
            if user in sensitive_attributes:
                sensitive_attr = torch.tensor([sensitive_attributes[user]], dtype=torch.long).to(device)
            else:
                sensitive_attr = torch.tensor([0], dtype=torch.long).to(device)
                
            # 獲取用戶特徵
            user_feat = torch.tensor(
                user_features[user] if user < len(user_features) else np.zeros_like(user_features[1]), 
                dtype=torch.float32
            ).unsqueeze(0).to(device)  # 添加批次維度
            
            # 創建候選項目集（所有項目）
            candidate_items = torch.arange(1, min(len(item_features), 10001)).long().to(device)  # 限制候選項目數量以節省計算
            
            # 計算用戶對每個候選項目的分數
            n_candidates = len(candidate_items)
            n_chunks = (n_candidates + 999) // 1000  # 每次處理1000個項目
            all_scores = []
            
            for i in range(n_chunks):
                start_idx = i * 1000
                end_idx = min((i + 1) * 1000, n_candidates)
                chunk_items = candidate_items[start_idx:end_idx]
                
                # 單獨獲取項目嵌入
                item_emb = model.item_embedding(chunk_items)
                
                # 計算基於用戶特徵的分數
                # 這是一個簡化的評分方法，僅用於排名
                user_emb = model.user_embedding(user_feat)
                scores = (user_emb.unsqueeze(1) * item_emb).sum(dim=2).squeeze(0)
                all_scores.append(scores)
            
            all_scores = torch.cat(all_scores, dim=0)
            
            # 排序項目
            _, rank_indices = torch.sort(all_scores, descending=True)
            rank_indices = rank_indices.cpu().numpy()
            
            # 找出目標項目在排序列表中的位置
            rank = np.where(candidate_items.cpu().numpy()[rank_indices] == target_item)[0]
            if len(rank) > 0:
                rank = rank[0]
            else:
                rank = len(rank_indices)  # 如果目標不在候選項目中，設置為最大長度
            
            # 計算評估指標
            for k in k_list:
                if rank < k:
                    metrics[f'HR@{k}'] += 1.0
                    metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)
                    
                    # 按敏感組計算（如果提供）
                    if sensitive_attributes and user in sensitive_attributes:
                        attr = sensitive_attributes[user]
                        metrics[f'HR@{k}_group{attr}'] += 1.0
                        metrics[f'NDCG@{k}_group{attr}'] += 1.0 / np.log2(rank + 2)
    
    # 計算平均值
    if valid_users > 0:
        for k in k_list:
            metrics[f'HR@{k}'] /= valid_users
            metrics[f'NDCG@{k}'] /= valid_users
    else:
        print("Warning: No valid users for evaluation!")
            
    # 計算每個敏感組的平均值
    if sensitive_attributes:
        group_counts = metrics.get('group_counts', {})
        for attr, count in group_counts.items():
            if count > 0:
                for k in k_list:
                    metrics[f'HR@{k}_group{attr}'] /= count
                    metrics[f'NDCG@{k}_group{attr}'] /= count
            else:
                print(f"Warning: No valid users in sensitive group {attr}!")
        
        # 計算組間差異
        for k in k_list:
            group_metrics = {attr: metrics.get(f'HR@{k}_group{attr}', 0.0) for attr in group_counts}
            if len(group_metrics) > 1:
                metrics[f'HR@{k}_disparity'] = max(group_metrics.values()) - min(group_metrics.values())
            
            group_metrics = {attr: metrics.get(f'NDCG@{k}_group{attr}', 0.0) for attr in group_counts}
            if len(group_metrics) > 1:
                metrics[f'NDCG@{k}_disparity'] = max(group_metrics.values()) - min(group_metrics.values())
    
    return metrics


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
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--fairness_lambda", type=float, default=0.1, help="Weight for the fairness loss term")
    parser.add_argument("--sensitive_embedding_dim", type=int, default=16, help="Dimension for sensitive attribute embeddings")
    parser.add_argument("--sensitive_feature_idx", type=int, default=0, help="Index of the context feature to use as sensitive attribute")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N epochs")

    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset
    dataset_files = data_partition(dataset_name)
    user_train, user_valid, user_test, usernum, itemnum = dataset_files

    if args.dataset == "Beauty":
        ItemFeatures = get_ItemDataBeauty(itemnum)
        UserFeatures = get_UserDataBeauty(usernum)
        CXTDict = load_data(f"./Data/CXTDictSasRec_{dataset_name}.dat")
    elif args.dataset == "Men":
        ItemFeatures = get_ItemDataMen(itemnum)
        UserFeatures = get_UserDataMen(usernum)
        CXTDict = load_data(f"./Data/CXTDictSasRec_{dataset_name}.dat")
    elif args.dataset == "Fashion":
        ItemFeatures = get_ItemDataFashion(itemnum)
        UserFeatures = get_UserDataFashion(usernum)
        CXTDict = load_data(f"./Data/CXTDictSasRec_{dataset_name}.dat")
    elif args.dataset == "Video_Games":
        ItemFeatures = get_ItemDataGames(itemnum)
        # 為Video_Games數據集創建用戶特徵（使用與其他數據集相同的方法）
        print(f"Creating identity matrix for UserFeatures for {dataset_name}.")
        UserFeatures = np.identity(usernum, dtype=np.int8)
        UserFeatures = np.vstack((np.zeros((1, usernum), dtype=np.int8), UserFeatures))
        CXTDict = load_data(f"./Data/CXTDictSasRec_{dataset_name}.dat")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    sensitive_attributes_map, num_sensitive_categories = load_sensitive_attributes(
        dataset_name, 
        user_train.keys(), 
        CXTDict, 
        default_attribute_index=0, 
        sensitive_feature_idx=args.sensitive_feature_idx
    )
    print(f"從上下文特徵中提取敏感屬性，類別數量: {num_sensitive_categories}")

    dataloader = get_dataloader(
        user_train,
        UserFeatures,
        itemnum,
        CXTDict,
        args.cxt_size,
        args.maxlen,
        args.batch_size,
        ItemFeatures,
        sensitive_attributes_map,
        shuffle=True
    )
    
    model = SequentialRecModel(
        usernum,
        itemnum,
        args,
        ItemFeatures.shape[1],
        UserFeatures.shape[1],
        args.cxt_size,
        num_sensitive_categories,
        args.sensitive_embedding_dim
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_emb if args.l2_emb > 0 else 0)

    best_valid_metric = 0.0
    for epoch in range(1, args.num_epochs + 1):
        rec_loss_epoch, fairness_loss_epoch, combined_loss_epoch = train_one_epoch(
            model, dataloader, optimizer, args.device, args.fairness_lambda
        )
        
        print(
            f"Epoch {epoch}, Rec Loss: {rec_loss_epoch:.4f}, Fairness Loss: {fairness_loss_epoch:.4f}, Combined Loss: {combined_loss_epoch:.4f}"
        )
        
        # 定期評估
        if epoch % args.eval_interval == 0:
            print("Evaluating on validation set...")
            metrics = evaluate(
                model, 
                user_valid, 
                ItemFeatures, 
                CXTDict, 
                UserFeatures, 
                args.device,
                sensitive_attributes=sensitive_attributes_map,
                maxlen=args.maxlen
            )
            
            # 打印主要指標
            print(f"NDCG@5: {metrics['NDCG@5']:.4f}, HR@5: {metrics['HR@5']:.4f}")
            print(f"NDCG@10: {metrics['NDCG@10']:.4f}, HR@10: {metrics['HR@10']:.4f}")
            
            # 如果有多個敏感組，打印組間差異
            if 'NDCG@10_disparity' in metrics:
                print(f"NDCG@10 Disparity: {metrics['NDCG@10_disparity']:.4f}")
                print(f"HR@10 Disparity: {metrics['HR@10_disparity']:.4f}")
            
            # 保存最佳模型
            current_metric = metrics['NDCG@10']
            if current_metric > best_valid_metric:
                best_valid_metric = current_metric
                # 可以在這裡保存模型
                print(f"New best model with NDCG@10: {best_valid_metric:.4f}")


if __name__ == "__main__":
    main()

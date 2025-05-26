import math
from typing import List
import numpy as np


def calculate_metrics(
    true_items_s, pred_items_s, pred_items_s_diff_gender, pred_items_ss_diff_age
):
    r"""
    pred_items_s: [[pred_items of user 1], ..., [pred_items of user N]]
    true_items_s: [[true_items of user 1], ..., [true_items of user N]]
    pred_items_s_diff_gender: [[pred_items of user 1 with different gender], ..., [pred_items of user N with different gender]]
    pred_items_ss_diff_age: [
                             [[pred_items of user 1 with age under 18], ..., [pred_items of user N with age under 18]],
                             [[pred_items of user 1 with age 18-24], ..., [pred_items of user N with age 18-24]],
                             [[pred_items of user 1 with age 25-34], ..., [pred_items of user N with age 25-34]],
                             [[pred_items of user 1 with age 35-44], ..., [pred_items of user N with age 35-44]],
                             [[pred_items of user 1 with age 45-49], ..., [pred_items of user N with age 45-49]],
                             [[pred_items of user 1 with age 50-55], ..., [pred_items of user N with age 50-55]],
                             [[pred_items of user 1 with age 56+], ..., [pred_items of user N with age 56+]]
                                                                                                           ]
    Fairness 的 evaluation metrics 參考自 CUFRL
    我將 \Delta DP (binary 時機率的差) 改成 Normalized Spearman Footrule Distance
    """
    N = len(pred_items_s)
    K = len(pred_items_s[0])

    # Vectorized NDCG@k
    def ndcg_at_k_batch(pred_items_s, true_items_s, k):
        ndcgs = np.zeros(N)
        for idx, (pred, true) in enumerate(zip(pred_items_s, true_items_s)):
            dcg = 0.0
            idcg = 0.0
            for i in range(k):
                if pred[i] in true:
                    dcg += 1 / math.log2(i + 2)
                idcg += 1 / math.log2(i + 2)
            ndcgs[idx] = dcg / idcg if idcg > 0 else 0.0
        return ndcgs.mean()

    # Hit rate@k
    def hit_rate_at_k_batch(pred_items_s, true_items_s, k):
        hits = 0
        for pred, true in zip(pred_items_s, true_items_s):
            if any(item in true for item in pred[:k]):
                hits += 1
        return hits / N

    # MRR@k
    def mrr_at_k_batch(pred_items_s, true_items_s, k):
        mrrs = np.zeros(N)
        for idx, (pred, true) in enumerate(zip(pred_items_s, true_items_s)):
            rr = 0.0
            for i in range(k):
                if pred[i] in true:
                    rr = 1.0 / (i + 1)
                    break
            mrrs[idx] = rr
        return mrrs.mean()

    def calculate_distance(R: List, S: List, k: int) -> float:
        """
        計算兩個 Top-k 列表之間的 Normalized Spearman Footrule Distance。
        """
        union = set(R) | set(S)
        rank_R = {item: i + 1 for i, item in enumerate(R)}
        rank_S = {item: i + 1 for i, item in enumerate(S)}
        def safe_log2(x):
            return math.log2(x + 1) if x != np.inf else np.inf
        F = sum(
            abs(
                1 / safe_log2(rank_R.get(item, np.inf))
                - 1 / safe_log2(rank_S.get(item, np.inf))
            )
            for item in union
        )
        norm = sum([1 / safe_log2(i + 1) for i in range(k)]) / 2
        return F / norm if norm > 0 else 0.0

    metrics = {}
    for k in [1, 5, 10, 20]:
        metrics[f'ndcg@{k}'] = ndcg_at_k_batch(pred_items_s, true_items_s, k=k)
    for k in [1, 5, 10, 20]:
        metrics[f'hit@{k}'] = hit_rate_at_k_batch(pred_items_s, true_items_s, k=k)
    for k in [1, 5, 10, 20]:
        metrics[f'mrr@{k}'] = mrr_at_k_batch(pred_items_s, true_items_s, k=k)

    # Vectorized fairness metrics
    # Gender
    dp_gender = np.mean([
        calculate_distance(items, items_diff_gender, k=K)
        for items, items_diff_gender in zip(pred_items_s, pred_items_s_diff_gender)
    ])
    metrics['dp_gender'] = dp_gender

    # Age (7 groups)
    dp_age = 0.0
    for user in range(N):
        dp_age_temp = 0.0
        for age in range(7):
            dp_age_temp += (
                calculate_distance(pred_items_ss_diff_age[age][user], pred_items_s[user], k=K)
                ** 2
            )
        dp_age += dp_age_temp ** 0.5
    metrics['dp_age'] = dp_age / N

    return metrics

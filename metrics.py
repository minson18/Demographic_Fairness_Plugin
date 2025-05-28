import math
from typing import List
import numpy as np
from typing import Optional


class Metrics:
    def __init__(self, all_scores: np.ndarray, true_items: np.ndarray):
        """
        all_scores: shape (num_users, num_items) - predicted scores/probabilities for all items
        true_items: shape (num_users,) - true item indices for each user
        """
        self.all_scores = all_scores
        self.true_items = true_items
        self.num_users = all_scores.shape[0]
        self.num_items = all_scores.shape[1]

    def _get_ranks(self):
        """
        For each user, get the rank (0-based) of the true item in the sorted predictions (descending).
        Returns: ranks: np.ndarray of shape (num_users,)
        """
        # argsort twice gives ranks
        sorted_indices = np.argsort(-self.all_scores, axis=1)  # descending
        # For each user, where is the true item in the sorted list?
        ranks = np.empty(self.num_users, dtype=np.int32)
        for i in range(self.num_users):
            ranks[i] = np.where(sorted_indices[i] == self.true_items[i])[0][0]
        return ranks

    def ndcg_at_k(self, k: int) -> float:
        """
        Compute NDCG@k for next-item prediction (one true item per user).
        """
        ranks = self._get_ranks()
        # Only count if rank < k
        dcg = (ranks < k) * (1.0 / np.log2(ranks + 2))  # +2 because rank is 0-based
        idcg = 1.0 / np.log2(2)  # always 1.0, since only one relevant item
        return np.mean(dcg / idcg)

    def mrr_at_k(self, k: int) -> float:
        """
        Compute MRR@k for next-item prediction (one true item per user).
        """
        ranks = self._get_ranks()
        rr = (ranks < k) * (1.0 / (ranks + 1))  # reciprocal rank if in top-k, else 0
        return np.mean(rr)

    def hit_at_k(self, k: int) -> float:
        """
        Compute Hit@k for next-item prediction (one true item per user).
        """
        ranks = self._get_ranks()
        hits = (ranks < k).astype(np.float32)
        return np.mean(hits)

    def distance_to(self, other: np.ndarray, k: Optional[int] = None) -> float:
        """
        Compute the mean Normalized Spearman Footrule Distance between this distribution and another (e.g., swapped feature predictions).
        If k is given, only top-k items are considered; otherwise, use all items.
        other: shape (num_users, num_items)
        Returns: mean distance over all users
        """
        assert other.shape == self.all_scores.shape
        n_users, n_items = self.all_scores.shape
        distances = np.zeros(n_users)
        for i in range(n_users):
            # Get ranking for both distributions
            if k is not None:
                R = np.argsort(-self.all_scores[i])[:k]
                S = np.argsort(-other[i])[:k]
            else:
                R = np.argsort(-self.all_scores[i])
                S = np.argsort(-other[i])
            # Build rank dicts
            rank_R = {item: idx + 1 for idx, item in enumerate(R)}
            rank_S = {item: idx + 1 for idx, item in enumerate(S)}
            union = set(R) | set(S)

            def safe_log2(x):
                return math.log2(x + 1) if x != np.inf else np.inf

            F = sum(
                abs(
                    1 / safe_log2(rank_R.get(item, np.inf))
                    - 1 / safe_log2(rank_S.get(item, np.inf))
                )
                for item in union
            )
            norm = sum([1 / safe_log2(i + 1) for i in range(len(R))]) / 2
            distances[i] = F / norm if norm > 0 else 0.0
        return np.mean(distances)

    def per_user_ndcg_at_k(self, k: int) -> np.ndarray:
        """
        Compute per-user NDCG@k for next-item prediction (one true item per user).
        Returns: np.ndarray of shape (num_users,)
        """
        ranks = self._get_ranks()
        dcg = (ranks < k) * (1.0 / np.log2(ranks + 2))
        idcg = 1.0 / np.log2(2)
        return dcg / idcg

    def delta_ndcg_at_k(self, other_scores: np.ndarray, k: int) -> float:
        """
        Compute the mean absolute difference in per-user NDCG@k between self and another score matrix.
        other_scores: shape (num_users, num_items)
        Returns: float
        """
        # Compute per-user NDCG@k for self
        ndcg_self = self.per_user_ndcg_at_k(k)
        # Compute per-user NDCG@k for other
        other_metrics = Metrics(other_scores, self.true_items)
        ndcg_other = other_metrics.per_user_ndcg_at_k(k)
        # Mean absolute difference
        return np.mean(np.abs(ndcg_self - ndcg_other))

    # Optionally, add more methods for fairness, etc.

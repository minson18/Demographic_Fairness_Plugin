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

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def distance_to(self, others: list, k: Optional[int] = None) -> float:
        """
        Compute the mean Euclidean distance between this distribution and multiple others.
        Each 'other' is a score matrix (num_users, num_items).
        Returns: mean distance over all users and all others.
        """
        self_probs = Metrics.softmax(self.all_scores)
        distances = []
        for other in others:
            assert other.shape == self.all_scores.shape
            other_probs = Metrics.softmax(other)
            if k is not None:
                # Only consider top-k items for each user
                topk_idx = np.argsort(self_probs, axis=1)[:, -k:]
                user_distances = []
                for i in range(self_probs.shape[0]):
                    idx = topk_idx[i]
                    diff = self_probs[i, idx] - other_probs[i, idx]
                    user_distances.append(np.linalg.norm(diff))
                distances.append(np.mean(user_distances))
            else:
                user_distances = np.linalg.norm(self_probs - other_probs, axis=1)
                distances.append(np.mean(user_distances))
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

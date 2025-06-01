import torch
import numpy as np
import pickle
from typing import List, Dict, Optional
from metrics import Metrics
import random
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Evaluator")


class Evaluator:
    def __init__(
        self,
        model,
        user_features: np.ndarray,
        item_features: np.ndarray,
        cxtdict: Dict,
        itemid2idx: Dict,
        device,
        user_valid: Optional[Dict] = None,
        user_test: Optional[Dict] = None,
    ):
        self.model = model
        self.user_features = user_features
        self.item_features = item_features
        self.cxtdict = cxtdict
        self.itemid2idx = itemid2idx
        self.device = device
        self.user_valid = user_valid
        self.user_test = user_test
        # Cache candidate items and tensors for efficiency
        self.candidate_items = list(self.itemid2idx.keys())
        self.candidate_indices = [
            self.itemid2idx[item] for item in self.candidate_items
        ]
        self.candidate_item_features = torch.tensor(
            self.item_features[self.candidate_indices],
            dtype=torch.float32,
            device=self.device,
        )
        # Cache user features as tensors for fast access
        self.user_features_tensor = torch.tensor(
            self.user_features, dtype=torch.float32, device=self.device
        )
        # Cache candidate contexts for all items (for get_top_k efficiency)
        self.candidate_contexts_cache = {}

        self.num_repeats = (
            3  # Number of times to repeat the evaluation in for fairness swap
        )

    def _build_sequence_tensors(self, train_seq: List[int], user: int) -> tuple:
        """
        Build sequence, feature, and context tensors for a user's history.
        """
        maxlen = self.model.maxlen
        # 1) map & truncate
        mapped = [self.itemid2idx[i] for i in train_seq if i in self.itemid2idx]
        if len(mapped) > maxlen - 1:
            mapped = mapped[-(maxlen - 1) :]
            orig_tail = [i for i in train_seq if i in self.itemid2idx][-(maxlen - 1) :]
        else:
            orig_tail = [i for i in train_seq if i in self.itemid2idx]
        # 2) build the index sequence (same as before)
        seq = np.zeros([maxlen], dtype=np.int32)
        seq[-(len(mapped) + 1) : -1] = mapped
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(
            0
        )
        # 3) build features (same as before)
        seq_feat = torch.zeros(
            (1, maxlen, self.item_features.shape[1]),
            dtype=torch.float32,
            device=self.device,
        )
        if mapped:
            seq_feat[0, -(len(mapped) + 1) : -1] = torch.tensor(
                self.item_features[mapped], dtype=torch.float32, device=self.device
            )
        # 4) correct context loop
        seqcxt = torch.zeros(
            (1, maxlen, self.model.cxt_size), dtype=torch.float32, device=self.device
        )
        for j, orig_item in enumerate(orig_tail):
            pos = -(len(mapped) + 1) + j
            seqcxt[0, pos, :] = torch.tensor(
                self.cxtdict.get((user, orig_item), np.zeros(self.model.cxt_size)),
                dtype=torch.float32,
                device=self.device,
            )
        return seq_tensor, seq_feat, seqcxt

    def _build_candidate_context(self, user: int) -> torch.Tensor:
        """
        Build context tensor for all candidate items for the given user.
        Returns: (num_candidates, cxt_size)
        """
        # Cache per-user candidate contexts if not already cached
        if user in self.candidate_contexts_cache:
            return self.candidate_contexts_cache[user]
        cxts = [
            self.cxtdict.get((user, item), np.zeros(self.model.cxt_size))
            for item in self.candidate_items
        ]
        cxts = np.array(cxts, dtype=np.float32)
        tensor_cxts = torch.tensor(cxts, dtype=torch.float32, device=self.device)
        self.candidate_contexts_cache[user] = tensor_cxts
        return tensor_cxts

    def build_batch_candidate_context(self, users, chunk_items):
        """
        Efficiently build context tensor for a batch of users and a chunk of items.
        Returns: (batch_size, chunk_size, cxt_size)
        """
        # Vectorized version: build all (user, item) pairs and gather
        batch_size = len(users)
        chunk_size = len(chunk_items)
        cxt_size = self.model.cxt_size
        user_arr = np.array(users).reshape(-1, 1)
        item_arr = np.array(chunk_items).reshape(1, -1)
        user_grid = np.broadcast_to(user_arr, (batch_size, chunk_size))
        item_grid = np.broadcast_to(item_arr, (batch_size, chunk_size))
        flat_user = user_grid.flatten()
        flat_item = item_grid.flatten()
        cxts = [
            self.cxtdict.get((u, i), np.zeros(cxt_size))
            for u, i in zip(flat_user, flat_item)
        ]
        cxts = np.array(cxts, dtype=np.float32).reshape(
            batch_size, chunk_size, cxt_size
        )
        return torch.tensor(cxts, dtype=torch.float32, device=self.device)

    def evaluate(
        self,
        user_train: Dict,
        user_eval: Dict,
        k: int = 20,
        batch_size: int = 128,
        candidate_chunk_size: int = 200,
        fairness_metrics: bool = True,
    ) -> dict:
        logger.info(
            f"Processing {len(user_eval)} users, batch_size={batch_size}, candidate_chunk_size={candidate_chunk_size}"
        )
        # Filter out users with None as their eval item
        filtered_users = [u for u in user_eval if user_eval[u] is not None]
        users = filtered_users
        true_items = np.array([self.itemid2idx[user_eval[u]] for u in users])
        all_scores = []
        for start in tqdm(
            range(0, len(users), batch_size), desc="Evaluating users", unit="user"
        ):
            end = min(start + batch_size, len(users))
            batch_users = users[start:end]
            batch_train_seqs = [user_train[u] for u in batch_users]
            batch_scores = self._get_all_scores_batch(
                batch_users,
                batch_train_seqs,
                candidate_chunk_size,
                swap_gender=None,
                swap_age=None,
                swap_occupation=None,
            )
            all_scores.append(batch_scores)
        all_scores = np.concatenate(all_scores, axis=0)
        metrics = Metrics(all_scores, true_items)
        results = {}
        for topk in [1, 5, 10, 20]:
            results[f"ndcg@{topk}"] = metrics.ndcg_at_k(topk)
            results[f"mrr@{topk}"] = metrics.mrr_at_k(topk)
            results[f"hit@{topk}"] = metrics.hit_at_k(topk)
        if fairness_metrics:
            # Gender fairness: only one swap needed since gender is binary
            swapped_genders = [1 - int(self.user_features[u - 1][0]) for u in users]
            batch_gender_scores = []
            for start in tqdm(
                range(0, len(users), batch_size),
                desc="Evaluating gender fairness",
                unit="user",
            ):
                end = min(start + batch_size, len(users))
                batch_users = users[start:end]
                batch_train_seqs = [user_train[u] for u in batch_users]
                batch_swapped_genders = [swapped_genders[i] for i in range(start, end)]
                batch_scores_gender = self._get_all_scores_batch(
                    batch_users,
                    batch_train_seqs,
                    candidate_chunk_size,
                    swap_gender=batch_swapped_genders,
                    swap_age=None,
                    swap_occupation=None,
                )
                batch_gender_scores.append(batch_scores_gender)
            all_scores_gender = [np.concatenate(batch_gender_scores, axis=0)]
            results["distance_gender"] = metrics.distance_to(all_scores_gender, k)
            # Delta NDCG for gender (no need to average, just use the single swap)
            for topk in [1, 5, 10, 20]:
                delta_ndcg_gender = metrics.delta_ndcg_at_k(all_scores_gender[0], topk)
                results[f"delta_ndcg_gender@{topk}"] = delta_ndcg_gender
            # Age fairness: multiple random swaps
            all_scores_age = []
            for repeat in tqdm(range(self.num_repeats), desc="Age fairness repeats"):
                swapped_ages = []
                for u in users:
                    original_age = int(self.user_features[u - 1][1])
                    choices = [ag for ag in range(7) if ag != original_age]
                    swapped_ages.append(np.random.choice(choices))
                batch_age_scores = []
                for start in range(0, len(users), batch_size):
                    end = min(start + batch_size, len(users))
                    batch_users = users[start:end]
                    batch_train_seqs = [user_train[u] for u in batch_users]
                    batch_swapped_ages = [swapped_ages[i] for i in range(start, end)]
                    batch_scores_age = self._get_all_scores_batch(
                        batch_users,
                        batch_train_seqs,
                        candidate_chunk_size,
                        swap_gender=None,
                        swap_age=batch_swapped_ages,
                        swap_occupation=None,
                    )
                    batch_age_scores.append(batch_scores_age)
                all_scores_age.append(np.concatenate(batch_age_scores, axis=0))
            results["distance_age"] = metrics.distance_to(all_scores_age, k)
            # Delta NDCG for age (average over repeats)
            for topk in [1, 5, 10, 20]:
                delta_ndcg_age = np.mean(
                    [
                        metrics.delta_ndcg_at_k(all_scores_age[repeat], topk)
                        for repeat in range(self.num_repeats)
                    ]
                )
                results[f"delta_ndcg_age@{topk}"] = delta_ndcg_age
            # Occupation fairness: multiple random swaps
            num_occ = self.user_features.shape[1] - 3
            all_scores_occ = []
            for repeat in tqdm(
                range(self.num_repeats), desc="Occupation fairness repeats"
            ):
                swapped_occs = []
                for u in users:
                    user_feat = self.user_features[u - 1]
                    original_occ_idx = np.argmax(user_feat[3 : 3 + num_occ])
                    choices = [i for i in range(num_occ) if i != original_occ_idx]
                    new_occ_idx = np.random.choice(choices)
                    swapped_occs.append(new_occ_idx)
                batch_occ_scores = []
                for start in range(0, len(users), batch_size):
                    end = min(start + batch_size, len(users))
                    batch_users = users[start:end]
                    batch_train_seqs = [user_train[u] for u in batch_users]
                    batch_swapped_occs = [swapped_occs[i] for i in range(start, end)]
                    batch_scores_occ = self._get_all_scores_batch(
                        batch_users,
                        batch_train_seqs,
                        candidate_chunk_size,
                        swap_gender=None,
                        swap_age=None,
                        swap_occupation=batch_swapped_occs,
                    )
                    batch_occ_scores.append(batch_scores_occ)
                all_scores_occ.append(np.concatenate(batch_occ_scores, axis=0))
            results["distance_occupation"] = metrics.distance_to(all_scores_occ, k)
            # Delta NDCG for occupation (average over repeats)
            for topk in [1, 5, 10, 20]:
                delta_ndcg_occ = np.mean(
                    [
                        metrics.delta_ndcg_at_k(all_scores_occ[repeat], topk)
                        for repeat in range(self.num_repeats)
                    ]
                )
                results[f"delta_ndcg_occupation@{topk}"] = delta_ndcg_occ
        return results

    def _get_all_scores_batch(
        self,
        users,
        train_seqs,
        candidate_chunk_size,
        swap_gender=None,
        swap_age=None,
        swap_occupation=None,
    ):
        batch_size = len(users)
        num_candidates = len(self.candidate_items)
        maxlen = self.model.maxlen
        user_indices = [u - 1 for u in users]
        user_feats = self.user_features_tensor[user_indices].clone()
        if swap_gender is not None:
            user_feats[:, 0] = torch.tensor(
                swap_gender, dtype=user_feats.dtype, device=self.device
            )
        if swap_age is not None:
            user_feats[:, 1] = torch.tensor(
                swap_age, dtype=user_feats.dtype, device=self.device
            )
        if swap_occupation is not None:
            num_occ = user_feats.shape[1] - 3
            for i, occ_idx in enumerate(swap_occupation):
                user_feats[i, 3 : 3 + num_occ] = 0
                user_feats[i, 3 + occ_idx] = 1
        # Use _build_sequence_tensors for each user to ensure correct context alignment
        seq_tensors = [
            self._build_sequence_tensors(seq, user)
            for seq, user in zip(train_seqs, users)
        ]
        seqs = torch.cat([t[0] for t in seq_tensors], dim=0)
        seq_feats = torch.cat([t[1] for t in seq_tensors], dim=0)
        seqcxts = torch.cat([t[2] for t in seq_tensors], dim=0)
        user_feats = user_feats.to(self.device)
        all_scores = []
        for chunk_start in range(0, num_candidates, candidate_chunk_size):
            chunk_end = min(chunk_start + candidate_chunk_size, num_candidates)
            chunk_indices = self.candidate_indices[chunk_start:chunk_end]
            chunk_items = self.candidate_items[chunk_start:chunk_end]
            chunk_item_features = self.candidate_item_features[chunk_start:chunk_end]
            candidate_contexts = self.build_batch_candidate_context(users, chunk_items)
            chunk_size = chunk_end - chunk_start
            pos = seqs.unsqueeze(1).repeat(1, chunk_size, 1).reshape(-1, maxlen)
            pos_feat = (
                seq_feats.unsqueeze(1)
                .repeat(1, chunk_size, 1, 1)
                .reshape(-1, maxlen, self.item_features.shape[1])
            )
            poscxt = (
                seqcxts.unsqueeze(1)
                .repeat(1, chunk_size, 1, 1)
                .reshape(-1, maxlen, self.model.cxt_size)
            )
            candidate_indices_tensor = torch.tensor(
                chunk_indices, dtype=torch.long, device=self.device
            )
            for i in range(batch_size):
                pos[i * chunk_size : (i + 1) * chunk_size, -1] = (
                    candidate_indices_tensor
                )
                pos_feat[i * chunk_size : (i + 1) * chunk_size, -1, :] = (
                    chunk_item_features
                )
                poscxt[i * chunk_size : (i + 1) * chunk_size, -1, :] = (
                    candidate_contexts[i]
                )
            user_feat_batch = (
                user_feats.unsqueeze(1)
                .repeat(1, chunk_size, 1)
                .reshape(-1, user_feats.shape[1])
            )
            seq_batch = seqs.unsqueeze(1).repeat(1, chunk_size, 1).reshape(-1, maxlen)
            seq_feat_batch = (
                seq_feats.unsqueeze(1)
                .repeat(1, chunk_size, 1, 1)
                .reshape(-1, maxlen, self.item_features.shape[1])
            )
            seqcxt_batch = (
                seqcxts.unsqueeze(1)
                .repeat(1, chunk_size, 1, 1)
                .reshape(-1, maxlen, self.model.cxt_size)
            )
            neg = pos
            neg_feat = pos_feat
            neg_cxt = poscxt
            with torch.no_grad():
                pos_logits, _, _ = self.model(
                    user_feat_batch,
                    seq_batch,
                    seq_feat_batch,
                    seqcxt_batch,
                    pos,
                    pos_feat,
                    poscxt,
                    neg,
                    neg_feat,
                    neg_cxt,
                )
                scores = pos_logits[:, -1].cpu().numpy().reshape(batch_size, chunk_size)
            all_scores.append(scores)
        all_scores = np.concatenate(all_scores, axis=1)
        return all_scores

    def sample_user_subset(
        self, user_eval: dict, percent: float = 0.1, seed: int = 42
    ) -> dict:
        """
        Return a random subset of user_eval containing percent of users.
        """
        users = list(user_eval.keys())
        random.seed(seed)
        sample_size = max(1, int(len(users) * percent))
        sampled_users = random.sample(users, sample_size)
        return {u: user_eval[u] for u in sampled_users}

    @staticmethod
    def print_metrics_table(metrics: dict):
        """
        Print a summary table of NDCG, Hit, MRR, and fairness metrics if present.
        """
        print("  |   k   | NDCG  | Hit   |  MRR  |", flush=True)
        print("  |-------|-------|-------|-------|", flush=True)
        for k in [1, 5, 10, 20]:
            print(
                f"  | {k:<5} | {metrics.get(f'ndcg@{k}', 0):.4f} | {metrics.get(f'hit@{k}', 0):.4f} | {metrics.get(f'mrr@{k}', 0):.4f} |",
                flush=True,
            )
        print("  |-------|-------|-------|-------|", flush=True)
        # Print fairness metrics if present
        if "distance_gender" in metrics:
            print(f"  Distance (gender): {metrics['distance_gender']:.4f}", flush=True)
        if "distance_age" in metrics:
            print(f"  Distance (age): {metrics['distance_age']:.4f}", flush=True)
        for k in [1, 5, 10, 20]:
            if f"delta_ndcg_gender@{k}" in metrics:
                print(
                    f"  Delta NDCG (gender)@{k}: {metrics[f'delta_ndcg_gender@{k}']:.4f}",
                    flush=True,
                )
            if f"delta_ndcg_age@{k}" in metrics:
                print(
                    f"  Delta NDCG (age)@{k}: {metrics[f'delta_ndcg_age@{k}']:.4f}",
                    flush=True,
                )
        if "distance_occupation" in metrics:
            print(
                f"  Distance (occupation): {metrics['distance_occupation']:.4f}",
                flush=True,
            )
        for k in [1, 5, 10, 20]:
            if f"delta_ndcg_occupation@{k}" in metrics:
                print(
                    f"  Delta NDCG (occupation)@{k}: {metrics[f'delta_ndcg_occupation@{k}']:.4f}",
                    flush=True,
                )

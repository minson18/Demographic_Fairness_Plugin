#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fairness-aware training ablation script.
Trains and tests models for a range of fairness_lambda values, with all other hyperparameters fixed.
Saves and prints a summary of test results for each lambda.
"""

import os
import sys
import json
import csv
import subprocess
from datetime import datetime
from train import train as train_func
import multiprocessing

# =====================
# Experiment config
# =====================
FIXED_PARAMS = {
    "dataset": "ml-1m",
    "epochs": 50,
    "batch_size": 256,
    "maxlen": 100,
    "lr": 0.0001,
    "hidden_units": 64,
    "num_blocks": 3,
    "dropout_rate": 0.3,
    "num_heads": 1,
    "cxt_size": 6,
    "device": "cuda",
    "sensitive_indices": "0,1,3",
}


# =====================
# Utility functions
# =====================
def load_json_metrics(path):
    """Load metrics from a JSON file, or return zeros if not found."""
    if os.path.exists(path):
        with open(path, "r") as f:
            metrics = json.load(f)
        return {
            "ndcg@20": metrics.get("ndcg@20", 0),
            "hit@20": metrics.get("hit@20", 0),
            "distance_gender": metrics.get("distance_gender", 0),
            "distance_age": metrics.get("distance_age", 0),
            "distance_occupation": metrics.get("distance_occupation", 0),
        }
    else:
        return {
            "ndcg@20": 0,
            "hit@20": 0,
            "distance_gender": 0,
            "distance_age": 0,
            "distance_occupation": 0,
        }


# =====================
# Core experiment logic
# =====================
def run_fair_training(fairness_lambda=0.1):
    """Run training with fairness constraints enabled and fixed params."""
    sys.argv = [
        "fair_train.py",
        f"--dataset={FIXED_PARAMS['dataset']}",
        f"--maxlen={FIXED_PARAMS['maxlen']}",
        f"--batch_size={FIXED_PARAMS['batch_size']}",
        f"--num_epochs={FIXED_PARAMS['epochs']}",
        "--use_fairness",
        f"--fairness_lambda={fairness_lambda}",
        f"--sensitive_indices={FIXED_PARAMS['sensitive_indices']}",
        f"--cxt_size={FIXED_PARAMS['cxt_size']}",
        f"--device={FIXED_PARAMS['device']}",
        f"--lr={FIXED_PARAMS['lr']}",
        f"--hidden_units={FIXED_PARAMS['hidden_units']}",
        f"--num_blocks={FIXED_PARAMS['num_blocks']}",
        f"--dropout_rate={FIXED_PARAMS['dropout_rate']}",
        f"--num_heads={FIXED_PARAMS['num_heads']}",
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"saved_models/fair_{FIXED_PARAMS['dataset']}_lambda{fairness_lambda}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    sys.argv.append(f"--model_dir={save_dir}")
    print(f"Starting fairness-aware training with lambda={fairness_lambda}")
    print(f"Fixed Params: {FIXED_PARAMS}")
    print(f"Results will be saved to: {save_dir}")
    train_func()
    return save_dir


def compare_fairness_lambdas(lambdas=[0.0, 0.1, 0.5, 1.0]):
    """
    For each lambda, train and test a model, then aggregate and save test results.
    """
    test_results = {}
    for fairness_lambda in lambdas:
        print(f"\n{'='*80}")
        print(f"Training with fairness_lambda = {fairness_lambda}")
        print(f"{'='*80}\n")
        save_dir = run_fair_training(fairness_lambda)
        # Load validation metrics (not used in summary, but could be)
        val_metrics = load_json_metrics(os.path.join(save_dir, "val_metrics.json"))
        # Run test.py as subprocess
        test_cmd = [
            sys.executable,
            "test.py",
            "--model_dir",
            save_dir,
            "--dataset",
            FIXED_PARAMS["dataset"],
        ]
        print(f"Running test.py for lambda={fairness_lambda}...")
        subprocess.run(test_cmd, check=True)
        # Load test metrics
        test_metrics = load_json_metrics(os.path.join(save_dir, "test_metrics.json"))
        test_results[fairness_lambda] = test_metrics
    # Save comparative test results to CSV (all metrics)
    # 1. Collect all metric keys
    all_keys = set(["lambda"])
    for metrics in test_results.values():
        all_keys.update(metrics.keys())
    all_keys = list(all_keys)
    # 2. Write CSV
    summary_path = f"saved_models/fairness_comparison_{FIXED_PARAMS['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_keys)
        writer.writeheader()
        for lambda_val, metrics in test_results.items():
            row = {"lambda": lambda_val}
            row.update(metrics)
            # Ensure all keys are present
            for k in all_keys:
                if k not in row:
                    row[k] = ""
            writer.writerow(row)
    # Print comparative test table
    print("\nTest Results Comparison:")
    print(
        f"{'Lambda':10} {'NDCG@20':10} {'Hit@20':10} {'Gender DP':10} {'Age DP':10} {'Occ DP':10}"
    )
    print(f"{'-'*70}")
    for lambda_val, metrics in test_results.items():
        print(
            f"{lambda_val:<10.2f} {metrics['ndcg@20']:<10.4f} {metrics['hit@20']:<10.4f} "
            f"{metrics['distance_gender']:<10.4f} {metrics['distance_age']:<10.4f} "
            f"{metrics['distance_occupation']:<10.4f}"
        )
    print(f"\nDetailed test results saved to: {summary_path}")


def train_and_test_lambda(fairness_lambda, save_dir, gpu_id=None):
    """
    Train and test for a single lambda, optionally on a specific GPU.
    Sets CUDA_VISIBLE_DEVICES if gpu_id is not None, runs training, then runs test.py, and returns test metrics.
    """
    import os
    import sys
    import subprocess

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.argv = [
        "fair_train.py",
        f"--dataset={FIXED_PARAMS['dataset']}",
        f"--maxlen={FIXED_PARAMS['maxlen']}",
        f"--batch_size={FIXED_PARAMS['batch_size']}",
        f"--num_epochs={FIXED_PARAMS['epochs']}",
        "--use_fairness",
        f"--fairness_lambda={fairness_lambda}",
        f"--sensitive_indices={FIXED_PARAMS['sensitive_indices']}",
        f"--cxt_size={FIXED_PARAMS['cxt_size']}",
        f"--device={FIXED_PARAMS['device']}",
        f"--lr={FIXED_PARAMS['lr']}",
        f"--hidden_units={FIXED_PARAMS['hidden_units']}",
        f"--num_blocks={FIXED_PARAMS['num_blocks']}",
        f"--dropout_rate={FIXED_PARAMS['dropout_rate']}",
        f"--num_heads={FIXED_PARAMS['num_heads']}",
        f"--model_dir={save_dir}",
    ]
    print(
        f"[GPU {gpu_id}] Training with lambda={fairness_lambda}, saving to {save_dir}"
    )
    train_func()
    test_cmd = [
        sys.executable,
        "test.py",
        "--model_dir",
        save_dir,
        "--dataset",
        FIXED_PARAMS["dataset"],
    ]
    print(f"[GPU {gpu_id}] Running test.py for lambda={fairness_lambda}")
    subprocess.run(test_cmd, check=True)
    test_metrics = load_json_metrics(os.path.join(save_dir, "test_metrics.json"))
    return fairness_lambda, test_metrics, save_dir


def run_fairness_lambdas_parallel(lambdas, gpu_ids=None):
    """
    Run each lambda as a separate process (parallel ablation), assigning GPUs if provided.
    """
    import time
    import multiprocessing

    test_results = {}
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    num_gpus = len(gpu_ids) if gpu_ids is not None else 0
    for i, fairness_lambda in enumerate(lambdas):
        gpu_id = gpu_ids[i % num_gpus] if num_gpus > 0 else None
        # Use microseconds for higher-resolution timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_dir = f"saved_models/fair_{FIXED_PARAMS['dataset']}_lambda{fairness_lambda}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        def worker(lam, save_dir, gpu_id, return_dict):
            try:
                lam_val, metrics, _ = train_and_test_lambda(lam, save_dir, gpu_id)
                return_dict[lam_val] = metrics
            except Exception as e:
                print(f"[GPU {gpu_id}] Error for lambda={lam}: {e}")
                return_dict[lam] = {"error": str(e)}

        p = multiprocessing.Process(
            target=worker, args=(fairness_lambda, save_dir, gpu_id, return_dict)
        )
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
    test_results = dict(return_dict)
    all_keys = set(["lambda"])
    for metrics in test_results.values():
        all_keys.update(metrics.keys())
    all_keys = list(all_keys)
    summary_path = f"saved_models/fairness_comparison_{FIXED_PARAMS['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_parallel.csv"
    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_keys)
        writer.writeheader()
        for lambda_val, metrics in test_results.items():
            row = {"lambda": lambda_val}
            row.update(metrics)
            for k in all_keys:
                if k not in row:
                    row[k] = ""
            writer.writerow(row)
    print("\nTest Results Comparison (Parallel):")
    print(f"{'Lambda':10} " + " ".join([f"{k:10}" for k in all_keys if k != "lambda"]))
    print(f"{'-'*70}")
    for lambda_val, metrics in test_results.items():
        row_str = f"{lambda_val:<10.2f} " + " ".join(
            [f"{metrics.get(k, '')!s:<10}" for k in all_keys if k != "lambda"]
        )
        print(row_str)
    print(f"\nDetailed test results saved to: {summary_path}")


# =====================
# CLI entrypoint
# =====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train recommendation models with controllable fairness (ablation on fairness_lambda)"
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        help="List of fairness lambda values to compare",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="List of GPU ids to use for parallel ablation (e.g. --gpus 0 1 2 3)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="(Internal) Model dir for single lambda run (used by multi-gpu launcher)",
    )
    args = parser.parse_args()

    # Always use parallel logic, assign GPUs if provided
    run_fairness_lambdas_parallel(args.lambdas, args.gpus)

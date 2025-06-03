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
import contextlib
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lambda_test")

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
    "alpha": 0.01,
    "beta": 0.25,
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


def train_and_test_lambda(fairness_lambda, save_dir, gpu_id=None):
    """
    Train and test for a single lambda, optionally on a specific GPU.
    Sets CUDA_VISIBLE_DEVICES if gpu_id is not None, runs training, then runs test.py, and returns test metrics.
    """

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Prepare sensitive indices as a list of strings
    sensitive_indices = [
        s.strip() for s in FIXED_PARAMS["sensitive_indices"].split(",")
    ]
    sys.argv = [
        "train.py",
        f"--dataset={FIXED_PARAMS['dataset']}",
        f"--maxlen={FIXED_PARAMS['maxlen']}",
        f"--batch_size={FIXED_PARAMS['batch_size']}",
        f"--num_epochs={FIXED_PARAMS['epochs']}",
        "--use_fairness",
        f"--fairness_lambda={fairness_lambda}",
        "--sensitive_indices",
        *sensitive_indices,
        f"--cxt_size={FIXED_PARAMS['cxt_size']}",
        f"--device={FIXED_PARAMS['device']}",
        f"--lr={FIXED_PARAMS['lr']}",
        f"--hidden_units={FIXED_PARAMS['hidden_units']}",
        f"--num_blocks={FIXED_PARAMS['num_blocks']}",
        f"--dropout_rate={FIXED_PARAMS['dropout_rate']}",
        f"--num_heads={FIXED_PARAMS['num_heads']}",
        f"--model_dir={save_dir}",
        f"--alpha={FIXED_PARAMS['alpha']}",
        f"--beta={FIXED_PARAMS['beta']}",
    ]
    logger.info(
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
        "--maxlen",
        str(FIXED_PARAMS["maxlen"]),
        "--use_fairness",
        "--fairness_lambda",
        str(fairness_lambda),
        "--sensitive_indices",
        *sensitive_indices,
        "--cxt_size",
        str(FIXED_PARAMS["cxt_size"]),
        "--device",
        FIXED_PARAMS["device"],
        "--hidden_units",
        str(FIXED_PARAMS["hidden_units"]),
        "--num_blocks",
        str(FIXED_PARAMS["num_blocks"]),
        "--dropout_rate",
        str(FIXED_PARAMS["dropout_rate"]),
        "--num_heads",
        str(FIXED_PARAMS["num_heads"]),
        "--alpha",
        str(FIXED_PARAMS["alpha"]),
        "--beta",
        str(FIXED_PARAMS["beta"]),
    ]
    logger.info(f"[GPU {gpu_id}] Running test.py for lambda={fairness_lambda}")
    subprocess.run(test_cmd, check=True)
    test_metrics = load_json_metrics(os.path.join(save_dir, "test_metrics.json"))
    return fairness_lambda, test_metrics, save_dir


def run_fairness_lambdas_parallel(lambdas, gpu_ids=None, model_dir=None):
    """
    Run each lambda as a separate process (parallel ablation), assigning GPUs if provided.
    Now logs stdout/stderr to job.log in each save_dir and prints progress/summary lines.
    Uses model_dir as the parent folder for all models/results.
    Ensures at most one process per GPU at a time.
    """

    if model_dir is None:
        model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)

    test_results = {}
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    num_gpus = len(gpu_ids) if gpu_ids is not None else 0
    total_jobs = len(lambdas)
    completed_jobs = 0
    save_dirs = []
    for i, fairness_lambda in enumerate(lambdas):
        gpu_id = gpu_ids[i % num_gpus] if num_gpus > 0 else None
        # Use microseconds for higher-resolution timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_dir = os.path.join(
            model_dir,
            f"fair_{FIXED_PARAMS['dataset']}_lambda{fairness_lambda}",
        )
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)

        def worker(lam, save_dir, gpu_id, return_dict):
            log_path = os.path.join(save_dir, "job.log")
            with open(log_path, "w") as log_file, contextlib.redirect_stdout(
                log_file
            ), contextlib.redirect_stderr(log_file):
                try:
                    logger.info(
                        f"[GPU {gpu_id}] Training with lambda={lam}, saving to {save_dir}"
                    )
                    lam_val, metrics, _ = train_and_test_lambda(lam, save_dir, gpu_id)
                    return_dict[lam_val] = metrics
                    logger.info(
                        f"[GPU {gpu_id}] Finished lambda={lam}. Metrics: {metrics}"
                    )
                except Exception as e:
                    logger.error(f"[GPU {gpu_id}] Error for lambda={lam}: {e}")
                    return_dict[lam] = {"error": str(e)}

        p = multiprocessing.Process(
            target=worker, args=(fairness_lambda, save_dir, gpu_id, return_dict)
        )
        jobs.append(p)
        p.start()
        # If we've launched num_gpus jobs, wait for all to finish before starting more
        if len(jobs) == num_gpus:
            for job in jobs:
                job.join()
                completed_jobs += 1
                logger.info(f"Progress: {completed_jobs}/{total_jobs} jobs completed.")
            jobs = []
    # Wait for any remaining jobs
    for job in jobs:
        job.join()
        completed_jobs += 1
        logger.info(f"Progress: {completed_jobs}/{total_jobs} jobs completed.")
    test_results = dict(return_dict)
    all_keys = set(["lambda"])
    for metrics in test_results.values():
        all_keys.update(metrics.keys())
    all_keys = list(all_keys)
    summary_path = os.path.join(
        model_dir,
        f"fairness_comparison_{FIXED_PARAMS['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_parallel.csv",
    )
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
    logger.info("\nTest Results Comparison (Parallel):")
    logger.info(
        f"{'Lambda':10} " + " ".join([f"{k:10}" for k in all_keys if k != "lambda"])
    )
    logger.info(f"{'-'*70}")
    for lambda_val, metrics in test_results.items():
        row_str = f"{lambda_val:<10.2f} " + " ".join(
            [f"{metrics.get(k, '')!s:<10}" for k in all_keys if k != "lambda"]
        )
        logger.info(row_str)
    logger.info(f"\nDetailed test results saved to: {summary_path}")
    logger.info("Log files for each run are saved as job.log in each model directory.")


# =====================
# CLI entrypoint
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train recommendation models with controllable fairness (ablation on fairness_lambda)"
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0, 0.01, 0.1, 0.25, 0.5, 1],
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
        help="Parent directory to save all models/results (default: saved_models)",
    )
    args = parser.parse_args()

    # Always use parallel logic, assign GPUs if provided
    run_fairness_lambdas_parallel(args.lambdas, args.gpus, args.model_dir)

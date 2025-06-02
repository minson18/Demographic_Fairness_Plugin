#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import argparse
from train import train as train_func
import json
import sys
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fair_train")


def run_fair_training(
    fairness_lambda=0.1, dataset="ml-1m", maxlen=100, batch_size=128, epochs=20
):
    """Run training with fairness constraints enabled"""
    # Build command line arguments
    sys.argv = [
        "fair_train.py",
        f"--dataset={dataset}",
        f"--maxlen={maxlen}",
        f"--batch_size={batch_size}",
        f"--num_epochs={epochs}",
        "--use_fairness",
        f"--fairness_lambda={fairness_lambda}",
        "--sensitive_indices=0,1",  # Gender and age are sensitive attributes
    ]

    # Create timestamped directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"saved_models/fair_{dataset}_lambda{fairness_lambda}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Add save_dir to args
    sys.argv.append(f"--model_dir={save_dir}")

    # Log experiment settings
    logger.info(f"Starting fairness-aware training with lambda={fairness_lambda}")
    logger.info(
        f"Dataset: {dataset}, Max length: {maxlen}, Batch size: {batch_size}, Epochs: {epochs}"
    )
    logger.info(f"Results will be saved to: {save_dir}")

    # Run training
    train_func()

    # Return the save directory for reference
    return save_dir


def compare_fairness_lambdas(lambdas=[0.0, 0.1, 0.5, 1.0], dataset="ml-1m", epochs=20):
    """Run multiple training runs with different fairness lambda values and compare results"""
    results = {}

    for fairness_lambda in lambdas:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training with fairness_lambda = {fairness_lambda}")
        logger.info(f"{'='*80}\n")

        save_dir = run_fair_training(fairness_lambda, dataset, epochs=epochs)

        # Load validation metrics
        metrics_path = os.path.join(save_dir, "val_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # Extract key metrics
            results[fairness_lambda] = {
                "ndcg@20": metrics.get("ndcg@20", 0),
                "hit@20": metrics.get("hit@20", 0),
                "distance_gender": metrics.get("distance_gender", 0),
                "distance_age": metrics.get("distance_age", 0),
                "distance_occupation": metrics.get("distance_occupation", 0),
            }

    # Save comparative results
    summary_path = f"saved_models/fairness_comparison_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print comparative table
    logger.info("\nResults Comparison:")
    logger.info(
        f"{'Lambda':10} {'NDCG@20':10} {'Hit@20':10} {'Gender DP':10} {'Age DP':10} {'Occ DP':10}"
    )
    logger.info(f"{'-'*70}")

    for lambda_val, metrics in results.items():
        logger.info(
            f"{lambda_val:<10.2f} {metrics['ndcg@20']:<10.4f} {metrics['hit@20']:<10.4f} "
            f"{metrics['distance_gender']:<10.4f} {metrics['distance_age']:<10.4f} "
            f"{metrics['distance_occupation']:<10.4f}"
        )

    logger.info(f"\nDetailed results saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train recommendation models with controllable fairness"
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.5, 1.0],
        help="List of fairness lambda values to compare",
    )
    parser.add_argument("--dataset", type=str, default="ml-1m", help="Dataset to use")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs for each run"
    )

    args = parser.parse_args()

    # Run comparative training
    compare_fairness_lambdas(args.lambdas, args.dataset, args.epochs)

import torch
import pickle
import os
from dataset import load_dataset
from model import CARCA
from evaluate import Evaluator
import argparse
import json
from train import convert_to_native
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test")


def load_split(split_name, out_dir):
    with open(os.path.join(out_dir, split_name), "rb") as f:
        return pickle.load(f)


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing best_model.pth and optionally val_metrics.json/config.json",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--candidate_chunk_size", type=int, default=200)
    # Add more hyperparameters as needed
    parser.add_argument("--hidden_units", type=int, default=90)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--l2_emb", type=float, default=0.0001)
    parser.add_argument("--cxt_size", type=int, default=6)
    parser.add_argument("--use_res", type=bool, default=True)
    args, _ = parser.parse_known_args()

    logger.info("Starting test.py")
    # If model_dir is provided, override model_path and try to load config/metrics
    if args.model_dir is not None:
        model_path = os.path.join(args.model_dir, "best_model.pth")
        logger.info(f"Using model_dir: {args.model_dir}")
        # Try to load val_metrics.json or config.json for hyperparameters
        config_path = os.path.join(args.model_dir, "val_metrics.json")
        if os.path.exists(config_path):
            logger.info(f"Found val_metrics.json at {config_path}")
            with open(config_path, "r") as f:
                val_metrics = json.load(f)
            # Optionally, set hyperparameters from val_metrics if stored
            # (Assumes you store them in val_metrics, or you can add a config.json)
        else:
            val_metrics = None
    elif args.model_path is not None:
        model_path = args.model_path
        logger.info(f"Using model_path: {model_path}")
        val_metrics = None
    else:
        model_path = os.path.join("saved_models", args.dataset, "best_model.pth")
        logger.info(f"Using default model_path: {model_path}")

    out_dir = f"Data/movielens_preprocessed"
    logger.info(f"Loading dataset: {args.dataset} (maxlen={args.maxlen})")
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
    ) = load_dataset(args.dataset, maxlen=args.maxlen)

    logger.info(f"Loading user_train_split and user_test from {out_dir}")
    user_train_split = load_split("user_train_split.pkl", out_dir)
    user_test = load_split("user_test.pkl", out_dir)

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    # Use hyperparameters from args (or optionally from val_metrics/config)
    model_args = argparse.Namespace(
        hidden_units=args.hidden_units,
        maxlen=args.maxlen,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        l2_emb=args.l2_emb,
        cxt_size=args.cxt_size,
        use_res=args.use_res,
    )
    model = CARCA(
        usernum,
        itemnum,
        model_args,
        item_features.shape[1],
        user_features.shape[1],
        cxtsize,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Starting evaluation on test set...")
    evaluator = Evaluator(
        model,
        user_features,
        item_features,
        cxtdict,
        itemid2idx,
        device,
    )
    metrics = evaluator.evaluate(
        user_train_split,
        user_test,
        k=20,
        batch_size=32,
        candidate_chunk_size=200,
    )
    logger.info("Test set metrics:")
    Evaluator.print_metrics_table(metrics)
    if "distance_gender" in metrics:
        logger.info(f"  Distance (gender): {metrics['distance_gender']:.4f}")
    if "distance_age" in metrics:
        logger.info(f"  Distance (age): {metrics['distance_age']:.4f}")
    # Save test metrics
    if args.model_dir is not None:
        save_path = os.path.join(args.model_dir, "test_metrics.json")
        with open(save_path, "w") as f:
            json.dump(convert_to_native(metrics), f, indent=2)  # Convert before dumping
        logger.info(f"Test metrics saved to {save_path}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    test()

import torch
import pickle
import os
from dataset import load_dataset
from model import CARCA
from evaluate import Evaluator, load_best_model
import argparse


def load_split(split_name, out_dir):
    with open(os.path.join(out_dir, split_name), "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--candidate_chunk_size", type=int, default=200)
    args = parser.parse_args()

    out_dir = f"Data/movielens_preprocessed"
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

    user_train_split = load_split("user_train_split.pkl", out_dir)
    user_test = load_split("user_test.pkl", out_dir)

    model_path = args.model_path or f"saved_models/{args.dataset}/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = CARCA(
        usernum,
        itemnum,
        argparse.Namespace(
            hidden_units=90,
            maxlen=args.maxlen,
            num_blocks=3,
            num_heads=1,
            dropout_rate=0.5,
            l2_emb=0.0001,
            cxt_size=cxtsize,
            use_res=True,
        ),
        item_features.shape[1],
        user_features.shape[1],
        cxtsize,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluator = Evaluator(
        model,
        user_features,
        item_features,
        cxtdict,
        itemid2idx,
        device,
    )
    metrics = evaluator.evaluate(user_train_split, user_test, k=20, batch_size=args.batch_size, candidate_chunk_size=args.candidate_chunk_size)
    print("Test set metrics:")
    print("  |   k   | NDCG  | Hit   |  MRR  |")
    print("  |-------|-------|-------|-------|")
    for k in [1, 5, 10, 20]:
        print(f"  | {k:<5} | {metrics[f'ndcg@{k}']:.4f} | {metrics[f'hit@{k}']:.4f} | {metrics[f'mrr@{k}']:.4f} |")
    print("  |-------|-------|-------|-------|")
    print(f"  DP_gender: {metrics['dp_gender']:.4f}")
    print(f"  DP_age: {metrics['dp_age']:.4f}")


if __name__ == "__main__":
    main()

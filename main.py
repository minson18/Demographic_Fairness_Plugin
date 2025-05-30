import argparse
from train import train
from test import test
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Run training, testing, or both.",
    )
    args, unknown = parser.parse_known_args()

    if args.mode == "train":
        sys.argv = [sys.argv[0]] + unknown
        train()
    elif args.mode == "test":
        sys.argv = [sys.argv[0]] + unknown
        test()
    elif args.mode == "both":
        sys.argv = [sys.argv[0]] + unknown
        train()
        sys.argv = [sys.argv[0]] + unknown
        test()


if __name__ == "__main__":
    main()

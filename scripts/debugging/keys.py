import argparse
import os


def main(args):
    os.environ["WANDB_API_KEY"] = args.wandb
    print(args.clearml_worker)
    print(args.wandb)
    print(os.getenv("WANDB_API_KEY"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clearml_worker', type=str)
    parser.add_argument('--wandb', type=str)
    args = parser.parse_args()
    main(args)

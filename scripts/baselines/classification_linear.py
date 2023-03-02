import logging


_logger = logging.getLogger(__name__)


def run_all(args):
    import numpy as np
    from datasets.config import DATASETS_DICT
    import wandb
    import clip
    from ood_detection.config import Config

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")

    for dname in datasets:
        if args.dname:
            if dname != args.dname:
                continue

        _logger.info(f"\t\t RUNNING {dname}")
        dset = DATASETS_DICT[dname]
        run = wandb.init(project=f"thesis-classification-linear_head",
                         entity="wandbefab",
                         name=dname,
                         config={'epochs': args.train_epochs,
                                 'lr': args.lr})

        from adapters.linear import full_linear_classification
        full_linear_classification(dset, clip_model, clip_transform, args.lr, args.train_epochs)

        run.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dname", type=str, default='all')
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max_split", type=int, default=2)

    args = parser.parse_args()

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_all(args)


if __name__=='__main__':
    main()
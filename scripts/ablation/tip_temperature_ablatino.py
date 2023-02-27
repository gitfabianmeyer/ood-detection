import logging

import numpy as np

_logger = logging.getLogger(__name__)


def run_all(args):
    import wandb
    from datasets.config import DATASETS_DICT

    datasets = list(DATASETS_DICT.keys())
    if args.split != 0:
        splits = np.array_split(datasets, 2)
        datasets = splits[args.split - 1]

    for dname in datasets:

        dset = DATASETS_DICT[dname]

        if args.dname != 'all':
            if dname != args.dname:
                continue

        run = wandb.init(project=f"thesis-ablation-tip-temperatures",
                         entity="wandbefab",
                         name=dname,
                         config=vars(args))

        run_single(dset, args)
        run.finish()


def run_single(dataset, args):
    import clip
    from ood_detection.config import Config
    from adapters.tip_adapter import get_cache_model
    from adapters.tip_adapter import get_dataset_features_with_split
    from adapters.tip_adapter import get_tip_adapter_train_set
    from adapters.tip_adapter import run_full_tip_from_features
    import numpy as np
    import wandb

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    clip_model.eval()
    train_set = get_tip_adapter_train_set(dataset, args.shots)
    _logger.info(f"len trainset: {len(train_set)}. Should be: {len(train_set.classes) * args.shots} (max)")

    # run everything on the val set first.
    _logger.info('----- VALIDATION PHASE-------')

    cache_keys, cache_values = get_cache_model(train_set, clip_model, augment_epochs=args.augment_epochs)

    val_features, val_labels, label_features, classes = get_dataset_features_with_split(dataset, clip_model,
                                                                                        clip_transform, 'val')

    test_features, test_labels, _, _ = get_dataset_features_with_split(dataset, clip_model,
                                                                       clip_transform,
                                                                       'test')

    temperatures = np.logspace(np.log2(0.001), np.log2(100), num=args.temperatures, base=2.0)
    for temp in temperatures:
        results = run_full_tip_from_features(cache_keys, cache_values, clip_model, args.eps, label_features,
                                             args.lr, temp, test_features, test_labels, args.train_epochs,
                                             train_set,
                                             val_features, val_labels)
        wandb.log(results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--temperatures", type=int, default=10)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--augment_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--dname", type=str, default='all')
    parser.add_argument("--split", type=int, default=0)

    args = parser.parse_args()
    print(vars(args))
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_all(args)


if __name__ == '__main__':
    main()

import logging

_logger = logging.getLogger(__name__)


def main():
    import clip
    import numpy as np
    import torch
    import wandb
    from datasets.config import DATASETS_DICT
    from metrics.distances import Distancer, MaximumMeanDiscrepancy
    from ood_detection.config import Config

    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")

    for dname in datasets:
        dset = DATASETS_DICT[dname]
        print(f"Running {dname}")
        run = wandb.init(project=f"thesis-mmd-{args.runs}",
                         entity="wandbefab",
                         name=dname,
                         tags=['distance',
                               'metrics'])

        clip_model, clip_transform = clip.load(Config.VISION_MODEL)
        dataset = dset(Config.DATAPATH,
                       split='val',
                       transform=clip_transform)

        _logger.info("Initializing distancer")
        from zeroshot.utils import FeatureDict
        feature_dict = FeatureDict(dataset)

        mmd = MaximumMeanDiscrepancy(feature_dict)
        results = []
        for i in range(args.runs):
            results.append(mmd.get_distance())
            _logger.info(results[-1])
        wandb.log({'mmd': np.mean(results),
                   'std': np.std(results)})

        run.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--split", type=int)
    parser.add_argument('--max_split', type=int)
    parser.add_argument('--runs', type=int)
    args = parser.parse_args()

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)

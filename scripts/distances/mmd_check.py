import logging



_logger = logging.getLogger(__name__)


# run this script 2 times: 1. with imagenet templates, 2 with custom templates

def main():
    import clip
    import numpy as np
    import torch
    import wandb
    from datasets.config import DATASETS_DICT
    from datasets.zoc_loader import IsolatedClasses
    from metrics.distances import Distancer, MaximumMeanDiscrepancy
    from ood_detection.config import Config

    for dname, dset in DATASETS_DICT.items():

        if dname!= 'imagenet':
            continue
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
            print(results[-1])
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

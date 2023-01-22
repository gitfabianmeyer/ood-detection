import logging

import clip
import numpy as np
import torch
import wandb
from datasets.config import DATASETS_DICT
from datasets.zoc_loader import IsolatedClasses
from metrics.distances import Distancer, MaximumMeanDiscrepancy
from ood_detection.config import Config

_logger = logging.getLogger(__name__)


# run this script 2 times: 1. with imagenet templates, 2 with custom templates

def main():
    for dname, dset in DATASETS_DICT.items():

        if dname!= 'imagenet':
            continue
        print(f"Running {dname}")
        run = wandb.init(project="thesis-mmd-100runs",
                         entity="wandbefab",
                         name=dname,
                         tags=['distance',
                               'metrics'])

        if dname == 'lsun':
            lsun = True
        else:
            lsun = False

        clip_model, clip_transform = clip.load(Config.VISION_MODEL)
        dataset = dset(Config.DATAPATH,
                       split='val',
                       transform=clip_transform)

        loaders = IsolatedClasses(dataset, batch_size=512, lsun=False)

        splits = 100  # run each exp 10 times
        id_split = Config.ID_SPLIT

        _logger.info("Initializing distancer")
        distancer = Distancer(isolated_classes=loaders,
                              clip_model=clip_model,
                              splits=splits,
                              id_split=id_split)

        mmd = MaximumMeanDiscrepancy(distancer.feature_dict)

        if dname == 'imagenet':
            print("Setting imagenet kernel size externally....")
            mmd.kernel_size = np.array(0.83247465, dtype=np.float32)
        print(f"kernel size: {mmd.kernel_size}")
        results = []
        for i in range(1):
            results.append(mmd.get_distance())
            print(results[-1])
        wandb.log({'mmd': np.mean(results),
                   'std': np.std(results)})

        run.finish()


if __name__ == '__main__':
    main()

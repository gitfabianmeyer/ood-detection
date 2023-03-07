import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np


import argparse
import logging

import clip

import wandb
from datasets.config import DATASETS_DICT
from ood_detection.config import Config

_logger = logging.getLogger(__name__)
splits = [(.4, .6), ]
clearml_model = False


def run_single_dataset_ood(dset, name, clip_model, clip_transform, id_classes=.6, runs=10):
    run = wandb.init(project="thesis-zoc_baseline_ten_classes_test_sets",
                     entity="wandbefab",
                     name=name,
                     config={"runs": runs,
                             "id_split": splits[0][0]})

    metrics = baseline_detector_no_temperature(dset,
                                               clip_model,
                                               clip_transform,
                                               Config.DEVICE,
                                               id_classes_split=id_classes,
                                               runs=runs)

    to_log = {}
    for key, metric in metrics.items():
        to_log[key] = np.mean(metric)
    wandb.log(to_log)
    run.finish()
    return True


def run_all(args):
    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    for dname, dset in DATASETS_DICT.items():

        # if dname != 'cifar10':
        #     print(f"Jumping over {dname}")
        #     continue

        _logger.info(f"---------------Running {dname}--------------")

        if dname == 'lsun':
            lsun = True

        else:
            lsun = False

        for split in splits:
            # perform zsoodd
            run_single_dataset_ood(dset=dset,
                                   name=dname,
                                   clip_model=clip_model,
                                   clip_transform=clip_transform,
                                   id_classes=split[0],
                                   runs=args.runs_ood)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_ood', type=int, default=5)

    args = parser.parse_args()
    run_all(args)

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from zoc.baseline import linear_layer_detector
import argparse
import logging

import clip

import torch
import wandb
from clearml import Task
from datasets.config import DATASETS_DICT
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config

_logger = logging.getLogger(__name__)
splits = [(.4, .6), ]
clearml_model = False
MODEL_PATH = "/home/fmeyer/ZOC/trained_models/COCO/ViT-B32/"
# MODEL_PATH = "/mnt/c/users/fmeyer/git/ood-detection/data/zoc/trained_models/COCO/"


def run_single_dataset_ood(dataset, clip_model, clip_transform, id_classes=.6, runs=5):
    dset = dataset(Config.DATAPATH,
                   split='test',
                   transform=None)
    labels = dset.classes
    id_classes = int(len(labels) * id_classes)
    ood_classes = len(labels) - id_classes

    # run = wandb.init(project="thesis-zoc_baseline_full_classes_val_sets",
    #                  entity="wandbefab",
    #                  name=name,
    #                  config={"runs": runs,
    #                          "id_split": splits[0][0]})

    metrics = linear_layer_detector(dataset, clip_model, clip_transform, id_classes, ood_classes, runs)
    print(metrics)
    return True


def run_all(args):
    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    for dname, dset in DATASETS_DICT.items():

        if dname not in ['cifar10', 'caltech cub']:
            print(f"Jumping over {dname}")
            continue

        _logger.info(f"---------------Running {dname}--------------")

        if dname == 'lsun':
            lsun = True

        else:
            lsun = False

        # shorted_classes = random.sample(dataset.classes, 10)
        # dataset.classes = shorted_classes
        for split in splits:
            # perform zsoodd
            run_single_dataset_ood(dset,
                                   clip_model=clip_model,
                                   clip_transform=clip_transform,
                                   id_classes=split[0],
                                   runs=args.runs_ood)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_ood', type=int, default=1)

    args = parser.parse_args()
    run_all(args)

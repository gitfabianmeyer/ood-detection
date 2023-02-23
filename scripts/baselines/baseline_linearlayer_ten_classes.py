import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import numpy as np
from zoc.utils import get_ablation_splits, get_split_specific_targets, fill_auc_lists, fill_f_acc_lists, \
    get_result_mean_dict



from zoc.baseline import linear_layer_detector, get_feature_weight_dict, FeatureSet
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


def run_single_dataset_ood(dataset, clip_model, clip_transform, id_classes=.4, runs=5):
    dset = dataset(Config.DATAPATH,
                   split='test',
                   transform=None)
    labels = dset.classes
    id_classes = int(10 * id_classes)
    ood_classes = 10 - id_classes
    metrics = linear_layer_detector(dataset, clip_model, clip_transform, id_classes, ood_classes, 5)
    run = wandb.init(project="thesis-zoc_baseline_linear_ten_classes_val_sets",
                     entity="wandbefab",
                     name=dset.name)
    wandb.log(metrics)
    run.finish()
    return True

def run_all(args):
    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    for dname, dset in DATASETS_DICT.items():

        # if dname not in ['cifar10', 'caltech cub']:
        #     print(f"Jumping over {dname}")
        #     continue

        _logger.info(f"---------------Running {dname}--------------")

        if dname == 'lsun':
            lsun = True

        else:
            lsun = False

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

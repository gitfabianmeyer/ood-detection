import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from zoc.baseline import baseline_detector
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


def run_single_dataset_ood(isolated_classes, name, clip_model, id_classes=.6, runs=5):
    labels = isolated_classes.labels
    id_classes = int(len(labels) * id_classes)
    ood_classes = len(labels) - id_classes

    run = wandb.init(project="thesis-zoc_baseline_full_classes",
                     entity="wandbefab",
                     name=name,
                     config={"runs": runs,
                             "id_split": splits[0][0]})

    metrics = baseline_detector(clip_model,
                                Config.DEVICE,
                                isolated_classes,
                                id_classes=id_classes,
                                ood_classes=ood_classes,
                                runs=1)
    for metric in metrics:
        wandb.log(metric)
    run.finish()
    return True


def run_all(args):
    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    for dname, dset in DATASETS_DICT.items():

        if dname != 'cifar10':
            print(f"Jumping over {dname}")
            continue

        _logger.info(f"Running {dname}...")

        if dname == 'lsun':
            lsun = True

        else:
            lsun = False

        dataset = dset(data_path=Config.DATAPATH,
                       train=False,
                       transform=clip_transform)

        # shorted_classes = random.sample(dataset.classes, 10)
        # dataset.classes = shorted_classes

        isolated_classes = IsolatedClasses(dataset,
                                           batch_size=512,
                                           lsun=lsun)

        for split in splits:
            # perform zsoodd
            run_single_dataset_ood(isolated_classes=isolated_classes,
                                   name=dname,
                                   clip_model=clip_model,
                                   id_classes=split[0],
                                   runs=args.runs_ood)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_ood', type=int, default=5)

    args = parser.parse_args()
    run_all(args)

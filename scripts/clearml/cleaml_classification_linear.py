import clip
import logging
import torch
import numpy
import wandb
import sklearn
import logging
import tqdm
import torchvision
import PIL
import imageio
import skimage
import sentencepiece
import os
import transformers
import pandas

from datasets.config import DATASETS_DICT
from clearml import Task
from adapters.linear import get_test_accuracy_from_dset
from adapters.linear import train_classification_head
from zeroshot.utils import get_feature_dict

import numpy as np
from ood_detection.config import Config
from zoc.baseline import FeatureSet

print("running clearml")
Task.add_requirements("git+https://github.com/gitfabianmeyer/ood-detection.git")
task = Task.init(project_name="ma_fmeyer", task_name=f"Classification-ViT-L")
task.execute_remotely('5e62040adb57476ea12e8593fa612186')
os.environ["WANDB_API_KEY"] = "a4628d0634b189525ab3a8352f52e2cd79f559b2"

_logger = logging.getLogger(__name__)


def run_all(args):
    _logger.info(f"Loading {args.vision}")
    clip_model, clip_transform = clip.load(args.vision)
    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")

    for dname in datasets:
        if args.dname:
            if dname != args.dname:
                _logger.info(f"Jumping {dname}")
                continue

        _logger.info(f"\t\t RUNNING {dname}")
        dset = DATASETS_DICT[dname]

        train_set = dset(Config.DATAPATH,
                         transform=clip_transform,
                         split='train')
        train_dict = get_feature_dict(train_set,
                                      clip_model)

        train = FeatureSet(train_dict, train_set.classes, train_set.class_to_idx)
        val_dict = get_feature_dict(dset(Config.DATAPATH,
                                         transform=clip_transform,
                                         split='val'),
                                    clip_model)

        val = FeatureSet(val_dict, train_set.classes, train_set.class_to_idx)

        feature_shape = clip_model.visual.output_dim
        output_shape = len(train_set.classes)

        max_accuracy = 0.
        for learning_rate in np.logspace(np.log2(0.0001), np.log2(0.01), args.lr, base=2):
            run = wandb.init(project=f"thesis-classification-linear_head-{dname}-{args.vision}",
                             entity="wandbefab",
                             name=str(learning_rate),
                             config={'epochs': args.train_epochs,
                                     'lr': args.lr})
            lr_acc, lr_classifier = train_classification_head(train,
                                                              val,
                                                              learning_rate,
                                                              args.train_epochs,
                                                              feature_shape,
                                                              output_shape,
                                                              True)
            if lr_acc > max_accuracy:
                max_accuracy = lr_acc
                best_classifier = lr_classifier
            run.finish()

        name = f"thesis-classification-linear_head-{dname}-{args.vision}"
        run = wandb.init(project=name,
                         entity="wandbefab",
                         name="test",
                         config={'epochs': args.train_epochs,
                                 'lr': args.lr})
        _logger.info("Getting test acc")
        test_set = dset(Config.DATAPATH,
                        transform=clip_transform,
                        split='test')

        test_dict = get_feature_dict(test_set, clip_model)
        test = FeatureSet(test_dict, test_set.classes, test_set.class_to_idx)
        acc = get_test_accuracy_from_dset(test, best_classifier)
        wandb.log({"test accuracy": acc})
        run.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=int, default=10)
    parser.add_argument("--dname", type=str, default=None)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max_split", type=int, default=0)
    parser.add_argument("--vision", type=str, default='ViT-L/14@336px')
    args = parser.parse_args()

    run_all(args)


if __name__ == '__main__':
    main()

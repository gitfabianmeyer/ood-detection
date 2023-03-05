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
from ood_detection.linear import get_test_accuracy_from_dset, train_classification_head
from zeroshot.utils import get_feature_sets_from_class, get_feature_dict_from_dataset

import numpy as np
from zeroshot.utils import FeatureSet

from zoc.detectors import train_log_reg_classifier

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

        if dname == 'imagenet' and args.clearml:
            from clearml import Dataset
            CLEARML_PATH = Dataset.get(dataset_name='tiny imagenet', dataset_project='Tiny Imagenet').get_local_copy()
            dset = DATASETS_DICT[dname]
            train_set = dset(CLEARML_PATH,
                             transform=clip_transform,
                             split='train',
                             clearml=True)
            train_dict = get_feature_dict_from_dataset(train_set,
                                                       clip_model)

            train = FeatureSet(train_dict, train_set.classes, train_set.class_to_idx)
            val_dict = get_feature_dict_from_dataset(dset(CLEARML_PATH,
                                                          transform=clip_transform,
                                                          split='val',
                                                          clearml=True),
                                                     clip_model)

            val = FeatureSet(val_dict, train_set.classes, train_set.class_to_idx)

            test_set = dset(CLEARML_PATH,
                            transform=clip_transform,
                            split='test',
                            clearml=True)

            test_dict = get_feature_dict_from_dataset(test_set, clip_model)
            test = FeatureSet(test_dict, test_set.classes, test_set.class_to_idx)

        else:
            from zeroshot.utils import get_feature_dict_from_class

            dset = DATASETS_DICT[dname]
            all_features = get_feature_sets_from_class(dset,
                                                       ['train', 'val', 'test'],
                                                       clip_model,
                                                       clip_transform)
            train = all_features['train']
            val = all_features["val"]
            test = all_features["test"]
        _logger.info(f"\t\t RUNNING {dname}")

        name = f"thesis-classification-logistic_head-large"
        run = wandb.init(project=name,
                         entity="wandbefab",
                         name=dname,
                         config={'num_c': args.cs})

        best_classifier = train_log_reg_classifier(train,
                                                   val,
                                                   args.cs)
        _logger.info("Getting test acc")
        test.features = test.features.cpu()
        acc = best_classifier.score(test.features, test.targets)
        wandb.log({"test accuracy": acc})
        run.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs", type=int, default=96)
    parser.add_argument("--dname", type=str, default=None)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max_split", type=int, default=0)
    parser.add_argument("--vision", type=str, default='ViT-L/14@336px')
    parser.add_argument("--clearml", type=str, default="True")
    args = parser.parse_args()

    run_all(args)


if __name__ == '__main__':
    main()

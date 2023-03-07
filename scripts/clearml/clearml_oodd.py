import os
import clip
import logging
import torch
import numpy as np
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
import logging
import wandb
from ood_detection.config import Config
from zoc.detectors import linear_layer_detector
from datasets.config import DATASETS_DICT
from zeroshot.utils import get_feature_dict_from_class, get_feature_dict_from_dataset, FeatureSet, FeatureDict
from clearml import Task




def run_all(args):
    _logger = logging.getLogger(__name__)

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")

    jumping = False
    if args.start_at:
        jumping = True
    for dname in datasets:

        if dname == args.start_at:
            jumping = False
        if jumping:
            _logger.info(f"Jumping {dname}")
            continue
        if args.dname:
            if dname != args.dname:
                _logger.info(f"Jumping {dname}")
                continue

        _logger.info(f"---------------Running {dname}--------------")
        run = wandb.init(project=f"thesis-ood-benchmark-{args.classifier_type}",
                         entity="wandbefab",
                         name=dname,
                         config={'type': args.classifier_type,
                                 'lr': args.lr})

        if dname == 'imagenet':
            from clearml import Dataset
            CLEARML_PATH = Dataset.get(dataset_name='tiny imagenet', dataset_project='Tiny Imagenet').get_local_copy()
            dset = DATASETS_DICT[dname]
            train_set = dset(CLEARML_PATH,
                             transform=clip_transform,
                             split='train',
                             clearml=True)
            train = FeatureDict(train_set,
                                clip_model)

            val = FeatureDict(dset(CLEARML_PATH,
                                   transform=clip_transform,
                                   split='val',
                                   clearml=True),
                              clip_model)

            test_set = dset(CLEARML_PATH,
                            transform=clip_transform,
                            split='test',
                            clearml=True)

            test = FeatureDict(test_set, clip_model)

        else:
            dset = DATASETS_DICT[dname]
            all_features = get_feature_dict_from_class(dset,
                                                       ['train', 'val', 'test'],
                                                       clip_model,
                                                       clip_transform)
            train = all_features['train']
            val = all_features["val"]
            test = all_features["test"]

        metrics = linear_layer_detector(train, val, test,
                                        args.runs,
                                        Config.ID_SPLIT,
                                        args.classifier_type,
                                        epochs=args.epochs,
                                        num_cs=96,
                                        learning_rate=args.lr)
        wandb.log(metrics)
        run.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument("--dname", type=str, default=None)
    parser.add_argument("--start_at", type=str, default="imagenet")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max_split", type=int, default=0)
    parser.add_argument('--classifier_type', type=str, default='logistic')
    parser.add_argument("--vision", type=str, default='ViT-L/14@336px')
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--clearml_worker', type=str)
    parser.add_argument('--wandb', type=str)
    args = parser.parse_args()

    Task.add_requirements("git+https://github.com/gitfabianmeyer/ood-detection.git")
    task = Task.init(project_name="ma_fmeyer", task_name=f"OOD Detection -ViT-L")
    task.execute_remotely(args.clearml_worker)
    os.environ["WANDB_API_KEY"] = args.wandb

    run_all(args)


if __name__ == '__main__':
    main()

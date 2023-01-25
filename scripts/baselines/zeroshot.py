import os
from collections import defaultdict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import clip
import numpy as np
import torch
from datasets import config
from datasets.config import DATASETS_DICT

import wandb
from datasets.cifar import OodCifar10
from ood_detection.config import Config
from ood_detection.classification_utils import accuracy, classify, get_dataset_features
from torch.utils.data import DataLoader
from tqdm import tqdm

from ood_detection.classification_utils import zeroshot_classifier


def get_std_mean(list_of_tensors):
    try:
        tensors = np.array(torch.stack(list_of_tensors))
        mean = np.mean(tensors)
        std = np.std(tensors)
    except Exception as e:
        mean, std = 0, 0
    return std, mean


def main():
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    clip_model.eval()
    for dname, dset in DATASETS_DICT.items():
        run = wandb.init(project="thesis-zsa-all-splits",
                         entity="wandbefab",
                         name=dname,
                         tags=['zeroshot',
                               'zsa'])

        results = {}
        for split in ['train', 'val', 'test']:
            print(f"\n\n----------------------------------- {dname}----------------------------------- ")

            dataset = dset(Config.DATAPATH,
                           split=split,
                           transform=clip_transform)

            dataloader = DataLoader(dataset,
                                    batch_size=512)

            features, targets = get_dataset_features(clip_model, dataloader)
            zsw = zeroshot_classifier(dataset.classes,
                                      dataset.templates,
                                      clip_model)

            results[f'{split} Acc'] = classify(features, zsw, targets)
        wandb.log(results)
        run.finish()


if __name__ == '__main__':
    main()

import time
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import PIL
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os
from ood_detection.classification_utils import full_classification
from ood_detection.config import Config
import torchvision

from datasets import caltech, flowers102, caltech_cub, svhn, pneumonia, imagenet, cifar, lsun, corruptions
from datasets import config as dconfig
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm

data_path = Config.DATAPATH
train = False
clip_model, transform_clip = clip.load(Config.VISION_MODEL)
corruption_dict = corruptions.Corruptions


class Checker:
    def __init__(self, id):
        self.id = id

    def __call__(self, sample):
        if self.id == 2:
            assert np.issubdtype(sample.dtype, np.integer)
            
        def check_for_numpy(sample: object) -> object:
            if np.issubdtype(sample.dtype, np.integer):
                assert sample.max() <= 255 and sample.min() >= 0
            elif np.issubdtype(sample.dtype, np.floating):
                assert sample.max() <= 1. and sample.min() >= 0.
            else:
                raise ValueError(f"WRONG DTYPE {sample.dtype}")
            return sample

        if isinstance(sample, np.ndarray):
            check_for_numpy(sample)
        else:
            img = np.asarray(sample)
            check_for_numpy(img)
        return sample


def get_transform(c, sev):
    transforms = []
    for trans in transform_clip.transforms[:-2]:
        transforms.append(trans)
    transforms.append(c(sev))
    transforms.append(ToTensor())
    transforms.append(Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))

    return Compose(transforms)


start = False
for dataset_name, dataset_obj in dconfig.DATASETS_DICT.items():
    if dataset_name == "dtd":
        start = True
    if start:
        print(dataset_name)
        for name, corr in corruption_dict.items():
            if name == "Glass Blur":
                print("PASSING GB")
                continue
            print(name)
            for severity in range(1, 2):
                print(f"{name}: {severity}")
                custom_transform = get_transform(corr, severity)
                dataset = dataset_obj(data_path, transform=custom_transform, train=train)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
                breaking = False

                for i, (image, target) in enumerate(dataloader):
                    if i % 200 == 0:
                        print(i)

                        if breaking:
                            i = 0
                            break
                        breaking = True

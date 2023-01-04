import argparse
import copy
import logging
import os

import numpy as np
import sentencepiece
import pycocotools
import clip
import torch
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig, BertTokenizer

from torch.optim import AdamW

from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from clearml import Dataset, Task

run_clearml = True

task = Task.init(project_name="ma_fmeyer", task_name="Test loading artifacts")


clip_image = True
if clip_image:
    artifact_task = Task.get_task(project_name='ma_fmeyer', task_name='clip_image_features')
else:
    artifact_task = Task.get_task(project_name='ma_fmeyer', task_name='bos_sentence_eos')

clip_backbone = 'ViT-B/32'
features = {}
for split in ['train', 'val']:

    if clip_image:
        features_path = 'clip_image_features_{}_{}.npy'.format(split,
                                                           clip_backbone)
    else:
        features_path = "bos_sentence_eos_{}_{}.npy".format(clip_backbone, split)

    artifact = artifact_task.artifacts[features_path].get_local_copy()

    artifact = np.load(artifact)

    print(features_path)
    print(type(artifact))
    print(artifact.files)
    arr = artifact[features_path]
    print(type(arr))
    print(arr.shape)
    print(40* "-")
    print()

    features[features_path] = arr

print(features.keys())


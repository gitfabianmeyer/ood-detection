import logging
import random

import numpy as np
import torch

from datasets.zoc_loader import IsolatedClasses

from ood_detection.config import Config
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

_logger = logging.getLogger(__name__)


@torch.no_grad()
def get_image_features_for_isolated_class_loader(loader, clip_model, stop_at=np.inf):
    features = []
    for images in loader:
        features.append(get_normalized_image_features(clip_model, images))
        if len(features) >= stop_at:
            print(f"Reached max {stop_at} for class {loader.name}")
            break

    return torch.cat(features)


@torch.no_grad()
def get_image_features_and_targets(loader, clip_model, stop_at=np.inf):
    features, targets = [], []
    for images, targets in loader:

        targets = targets.to(Config.DEVICE)
        targets.append(targets)

        features.append(get_normalized_image_features(clip_model, images))

        if len(features) >= stop_at:
            print(f"Reached max {stop_at} for class {loader.name}")
            break

    return torch.cat(features), torch.cat(targets)


@torch.no_grad()
def get_normalized_image_features(clip_model, images):
    images = images.to(Config.DEVICE)
    image_features = clip_model.encode_image(images)
    image_features /= image_features.norm(dim=1, keepdim=True)
    return image_features


@torch.no_grad()
def get_feature_weight_dict(isolated_classes, clip_model):
    weights_dict = {}
    _logger.info("generating features from isolated classes")
    for cls in isolated_classes.classes:
        loader = isolated_classes[cls]

        features = get_image_features_for_isolated_class_loader(loader, clip_model)
        weights_dict[cls] = features.half()

    return weights_dict


def get_feature_dict(dataset, clip_model):
    isolated_classes = IsolatedClasses(dataset, batch_size=512)
    # return {random.randint(1, 1000): torch.rand((random.randint(1, 5), 4)) for i in range(3)}
    return get_feature_weight_dict(isolated_classes, clip_model)


class FeatureDict:
    def __init__(self, dataset, clip_model):

        self.feature_dict, self.classes = self.init_feature_dict(dataset, clip_model)

    def __len__(self):
        return len(self.classes)

    def get_features(self):
        return torch.cat(list(self.feature_dict.values()))

    def items(self):
        return self.feature_dict.items()

    def keys(self):
        return self.feature_dict.keys()

    def values(self):
        return self.feature_dict.values()

    def __getitem__(self, item):
        return self.feature_dict[item]

    def init_feature_dict(self, dataset, clip_model):
        if isinstance(dataset, dict):
            return dataset, list(dataset.keys())
        elif isinstance(dataset, Dataset) or isinstance(dataset, ImageFolder):
            return get_feature_dict(dataset, clip_model), dataset.classes

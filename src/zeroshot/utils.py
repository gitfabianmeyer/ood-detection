import logging
from typing import List

import numpy as np
import torch
from datasets.classnames import base_template

from datasets.zoc_loader import IsolatedClasses
from ood_detection.classification_utils import zeroshot_classifier

from ood_detection.config import Config
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

_logger = logging.getLogger(__name__)


class FeatureDict:
    def __init__(self, dataset, clip_model=None):

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
            return get_feature_dict_from_dataset(dataset, clip_model), dataset.classes


class FeatureSet(Dataset):
    def __init__(self, feature_dict, labels, class_to_idx_mapping):
        self.labels = labels
        self.class_to_idx_mapping = class_to_idx_mapping
        self.features, self.targets = self.get_features_labels(feature_dict)
        self.features_dim = self.features[0].shape[0]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], int(self.targets[idx])

    def get_features_labels(self, feature_dict):
        features, targets = [], []
        for label in self.labels:
            feats = feature_dict[label]
            targets.extend([self.class_to_idx_mapping[label]] * len(feats))
            features.append(feats)
        return torch.cat(features), torch.Tensor(targets)


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


def get_feature_dict_from_dataset(dataset, clip_model):
    isolated_classes = IsolatedClasses(dataset, batch_size=512)
    # return {random.randint(1, 1000): torch.rand((random.randint(1, 5), 4)) for i in range(3)}
    return get_feature_weight_dict_from_isolated_from_isolated_classes(isolated_classes, clip_model)


def get_feature_dict_from_class(dset, splits: List, clip_model, transform):
    feature_dicts = {}
    for split in splits:
        dataset = dset(Config.DATAPATH,
                       transform=transform,
                       split=split)
        feature_dicts[split] = FeatureDict(dataset, clip_model)

    return feature_dicts


def get_ablation_split_classes(num_id_classes, split):
    seen_labels = split[:num_id_classes]
    template = base_template[0]
    seen_discriptions = [template.format(label) for label in seen_labels]

    unseen_labels = split[num_id_classes:]
    return seen_discriptions, seen_labels, unseen_labels


def get_feature_and_class_weight_dict_from_dset(dset, clip_model, clip_transform, split='test'):
    isolated_classes = IsolatedClasses(dset(Config.DATAPATH,
                                            transform=clip_transform,
                                            split=split),
                                       batch_size=512)

    return get_feature_and_class_weight_dict(isolated_classes, clip_model)


def get_feature_and_class_weight_dict(isolated_classes_fast_loader, clip_model):
    _logger.info("Creating feature and class weight dicts")
    feature_weight_dict = get_feature_weight_dict_from_isolated_from_isolated_classes(isolated_classes_fast_loader,
                                                                                      clip_model)
    classes_weight_dict = get_zeroshot_weight_dict_from_isolated_classes(isolated_classes_fast_loader, clip_model)

    return FeatureDict(feature_weight_dict), FeatureDict(classes_weight_dict)


def get_zeroshot_weight_dict_from_isolated_classes(isolated_classes, clip_model):
    weights_dict = {}

    if isinstance(isolated_classes, IsolatedClasses):
        weights = zeroshot_classifier(isolated_classes.classes, isolated_classes.templates, clip_model)

        for classname, weight in zip(isolated_classes.classes, weights):
            weights_dict[classname] = weight

    return FeatureDict(weights_dict)


@torch.no_grad()
def get_feature_weight_dict_from_isolated_from_isolated_classes(isolated_classes, clip_model):
    weights_dict = {}
    _logger.info("generating features from isolated classes")
    for cls in isolated_classes.classes:
        loader = isolated_classes[cls]

        features = get_image_features_for_isolated_class_loader(loader, clip_model)
        weights_dict[cls] = features.half()

    return weights_dict


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
def get_normalized_image_features(clip_model, images):
    images = images.to(Config.DEVICE)
    image_features = clip_model.encode_image(images)
    image_features /= image_features.norm(dim=1, keepdim=True)
    return image_features

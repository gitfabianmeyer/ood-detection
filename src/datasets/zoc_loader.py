import io
from typing import Tuple, Any

import numpy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class IsolatedLsunClass(Dataset):
    def __init__(self, dataset, class_label=None):
        assert class_label, 'a semantic label must be specified'
        self.transform = dataset.transform
        self.class_label = class_label
        class_mask = np.array(dataset.targets) == dataset.class_to_idx[class_label]
        self.db = dataset.dbs[dataset.class_to_idx[class_label]]
        self.targets = torch.tensor(dataset.targets[class_mask])
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, _ = self.db[index]
        return img

    @property
    def name(self):
        return self.class_label


class IsolatedClass(Dataset):
    def __init__(self, dataset, class_label=None):
        assert class_label, 'a semantic label must be specified'
        self.transform = dataset.transform
        self.class_label = class_label
        class_mask = np.array(dataset.targets) == dataset.class_to_idx[class_label]
        self.data = [dataset.data[i] for i in range(len(dataset.data)) if class_mask[i]]
        self.targets = torch.tensor(dataset.targets[class_mask])
        self.class_to_idx = dataset.class_to_idx

    def __getitem__(self, idx):
        image_file = self.data[idx]
        if isinstance(image_file, np.ndarray):
            if np.logical_and(image_file >= 0, image_file <= 1).all() and np.issubdtype(image_file.dtype, np.floating):
                image_file = (image_file * 255).astype(np.uint8)
            elif np.logical_and(image_file >= 0, image_file <= 255).all():

                image_file = image_file.astype(np.uint8)
            img = Image.fromarray(image_file)
        else:
            # works for most datasets
            img = Image.open(image_file)
        return self.transform(img.convert('RGB'))

    def __len__(self):
        return len(self.data)

    @property
    def name(self):
        return self.class_label


def single_isolated_class_loader(full_dataset, batch_size=1, lsun=False):
    loaders_dict = {}
    labels = full_dataset.classes
    for label in labels:
        if lsun:
            dataset = IsolatedLsunClass(full_dataset, label)
        else:
            dataset = IsolatedClass(full_dataset, label)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
        loaders_dict[label] = loader

    return loaders_dict


def get_loader(dataset):
    return DataLoader(dataset, batch_size=128)

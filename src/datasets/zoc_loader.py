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
            try:
                img = Image.fromarray(image_file)
            except TypeError:
                # svhn
                img = Image.fromarray(np.transpose(image_file, (1, 2, 0)))
        elif isinstance(image_file, torch.Tensor):
            # mnist & fashionmnist
            img = Image.fromarray(image_file.numpy(), mode="L")
        else:
            # works for most datasets
            img = Image.open(image_file)
        return self.transform(img.convert('RGB'))

    def __len__(self):
        return len(self.data)

    @property
    def name(self):
        return self.class_label


class IsolatedClasses:
    def __init__(self, dataset, batch_size=1, lsun=False):
        self.loaders_dict = {}
        self.templates = dataset.templates
        self.lsun = lsun
        self.batch_size = batch_size

        self.classes = dataset.classes
        self.fill_loaders_dict(dataset)

    def fill_loaders_dict(self, full_dataset):
        for label in self.classes:
            if self.lsun:
                dset = IsolatedLsunClass(full_dataset, label)
            else:
                dset = IsolatedClass(full_dataset, label)

            loader = DataLoader(dataset=dset, batch_size=self.batch_size, num_workers=4)
            self.loaders_dict[label] = loader

    def keys(self):
        return self.loaders_dict.keys()

    def values(self):
        return self.loaders_dict.values()

    def items(self):
        return self.loaders_dict.items()

    def __iter__(self):
        return iter(self.loaders_dict)

    def __getitem__(self, key):
        return self.loaders_dict[key]

    def __setitem__(self, key, value):
        self.loaders_dict[key] = value

    def __repr__(self):
        return repr(self.loaders_dict)

    def __contains__(self, item):
        return item in self.loaders_dict

    def __iter__(self):
        return iter(self.loaders_dict)


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

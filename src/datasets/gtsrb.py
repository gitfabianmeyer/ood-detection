import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from typing import Tuple, Any

import PIL
from sklearn.model_selection import train_test_split
import logging

import clip
import numpy as np
import torchvision
from datasets.classnames import gtsrb_classes, gtsrb_templates
from ood_detection.config import Config

from metrics.distances import get_distances_for_dataset, run_full_distances

logging.basicConfig(level=logging.INFO)


class OodGTSRB(torchvision.datasets.GTSRB):
    def __init__(self, data_path, transform, split, templates=None):
        super().__init__(data_path,
                         split='train' if split == 'train' or split == 'val' else 'test',
                         download=True,
                         transform=transform)
        self.split = split
        self.classes = gtsrb_classes
        self.templates = templates if templates else gtsrb_templates
        self.class_to_idx = dict(zip(self.classes, list(range(len(self.classes)))))
        self.idx_to_class = dict(zip(list(range(len(self.classes))), self.classes))
        self.data, self.targets = zip(*self._samples)
        self.targets = np.array(self.targets)
        self.set_split()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path = self.data[index]
        sample = PIL.Image.open(path).convert("RGB")

        target = self.targets[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def set_split(self):
        if self.split == 'val':
            _, self.data, _, self.targets = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)
        elif self.split == 'train':
            self.data, _, self.targets, _ = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def name(self):
        return 'gtsrb'


def main():
    name = "gtrsb"
    dataset = OodGTSRB
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()

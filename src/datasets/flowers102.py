import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging

import clip
import numpy as np
import torchvision.datasets

from datasets.classnames import flowers102_classes, flowers102_templates
from ood_detection.config import Config
from metrics.distances import get_distances_for_dataset, run_full_distances

logging.basicConfig(level=logging.INFO)


class OodFlowers102(torchvision.datasets.Flowers102):
    def __init__(self, data_path, transform, split, templates=None):
        super().__init__(data_path,
                         transform=transform,
                         split=split,
                         download=True)
        self.classes = flowers102_classes
        self.templates = templates if templates else flowers102_templates
        self.data = self._image_files
        self.targets = np.array(self._labels)
        self.class_to_idx = {cls: i for (i, cls) in enumerate(self.classes)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.data)

    @property
    def name(self):
        return 'flowers102'


def main():
    name = "Flowers102"
    dataset = OodFlowers102
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()

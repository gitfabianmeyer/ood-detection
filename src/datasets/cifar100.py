import logging

import clip
import numpy as np
import torchvision
from datasets.classnames import cifar_templates
from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config

from metrics.distances import run_full_distances
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


class OodCifar100(torchvision.datasets.CIFAR100):
    def __init__(self, data_path, transform, split, templates=None):
        super(OodCifar100, self).__init__(root=data_path,
                                          transform=transform,
                                          train=True if split == 'train' or split == 'val' else False,
                                          download=True
                                          )
        self.split = split
        self.targets = np.array(self.targets)
        self.templates = templates if templates else cifar_templates
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}
        self.set_split()

    def set_split(self):
        if self.split == 'val':
            _, self.data, _, self.targets = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)
        elif self.split == 'train':
            self.data, _, self.targets, _ = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)
    @property
    def name(self):
        return 'cifar100'

def main():
    name = "cifar100"
    dataset = OodCifar100
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()

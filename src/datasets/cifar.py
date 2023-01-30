import logging
import numpy as np
import torchvision.datasets
from datasets.classnames import cifar_templates

from metrics.distances import run_full_distances
from ood_detection.config import Config
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


class OodCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, data_path, transform, split, templates=None):
        super(OodCifar10, self).__init__(root=data_path,
                                         transform=transform,
                                         train=True if split == 'train' or split == 'val' else False,
                                         download=True
                                         )
        self.split = split
        self.targets = np.array(self.targets)
        self.templates = templates if templates else cifar_templates
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}
        self.set_split()
        self.name = 'cifar10'

    def set_split(self):
        if self.split == 'val':
            _, self.data, _, self.targets = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)
        elif self.split == 'train':
            self.data, _, self.targets, _ = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)


def main():
    dataset = OodCifar10
    run_full_distances(dataset.name, dataset, lsun=False)


if __name__ == '__main__':
    main()

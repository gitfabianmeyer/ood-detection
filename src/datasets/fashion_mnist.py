import os

from ood_detection.config import Config
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging

import torchvision

from datasets.classnames import mnist_templates
from metrics.distances import run_full_distances

logging.basicConfig(level=logging.INFO)


class OodFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, data_path, transform, split, templates=None):
        super(OodFashionMNIST, self).__init__(root=data_path,
                                              transform=transform,
                                              download=True,
                                              train=True if split == 'train' or split == 'val' else False,)
        self.templates = templates if templates else mnist_templates
        self.split = split
        self.set_split()

    def set_split(self):
        if self.split == 'val':
            _, self.data, _, self.targets = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)
        elif self.train == 'train':
            self.data, _, self.targets, _ = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)
    @property
    def name(self):
        return 'fashion mnist'

def main():
    name = "Fashion MNIST"
    dataset = OodFashionMNIST
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging

import torchvision

from datasets.classnames import mnist_templates
from metrics.distances import run_full_distances

logging.basicConfig(level=logging.INFO)


class OodFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, data_path, transform, train, templates=None):
        super(OodFashionMNIST, self).__init__(root=data_path,
                                              transform=transform,
                                              download=True,
                                              train=train)
        self.templates = templates if templates else mnist_templates


def main():
    name = "Fashion MNIST"
    dataset = OodFashionMNIST
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()

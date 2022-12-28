import logging

import clip
import torchvision.datasets
from datasets.classnames import mnist_templates

from metrics.distances import get_distances_for_dataset, run_full_distances
from ood_detection.config import Config

logging.basicConfig(level=logging.INFO)


class OodMNIST(torchvision.datasets.MNIST):
    def __init__(self, data_path, transform, train, templates=None):
        super(OodMNIST, self).__init__(root=data_path,
                                       transform=transform,
                                       download=True,
                                       train=train)

        self.templates = templates if templates else mnist_templates
        self.classes = [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
        ]
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}


def main():
    name = "MNIST"
    dataset = OodMNIST
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()

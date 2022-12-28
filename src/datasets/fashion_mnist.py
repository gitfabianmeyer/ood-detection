import logging

import clip
import torchvision

from datasets.classnames import mnist_templates
from metrics.distances import get_distances_for_dataset, get_corruption_metrics, run_full_distances
from ood_detection.config import Config

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

import logging

import clip
import numpy as np
import torchvision
from datasets.classnames import cifar_templates
from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config

from metrics.distances import run_full_distances

logging.basicConfig(level=logging.INFO)


class OodCifar100(torchvision.datasets.CIFAR100):
    def __init__(self, data_path, transform, train, templates=None):
        super(OodCifar100, self).__init__(root=data_path,
                                          transform=transform,
                                          train=train,
                                          download=True)
        self.targets = np.array(self.targets)
        self.templates = templates if templates else cifar_templates


def main():
    name = "cifar100"
    dataset = OodCifar100
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()


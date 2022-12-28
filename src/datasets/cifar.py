import logging
import numpy as np
import torchvision.datasets
from datasets.classnames import cifar_templates

from metrics.distances import run_full_distances

logging.basicConfig(level=logging.INFO)


class OodCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, data_path, transform, train, templates=None):
        super(OodCifar10, self).__init__(root=data_path,
                                         transform=transform,
                                         train=train,
                                         download=True
                                         )
        self.targets = np.array(self.targets)
        self.templates = templates if templates else cifar_templates
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}


def main():
    name = "cifar10"
    dataset = OodCifar10
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()

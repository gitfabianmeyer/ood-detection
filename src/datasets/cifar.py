import logging
import clip
import numpy as np
import torchvision.datasets
from datasets.classnames import cifar_templates
from ood_detection.config import Config

from metrics.distances import get_distances_for_dataset

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


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    cifar = OodCifar10(data_path, transform, train)
    get_distances_for_dataset(cifar, clip_model, "cifar10")


if __name__ == '__main__':
    main()

import logging

import clip
import numpy as np
import torchvision
from datasets.classnames import cifar_templates
from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config

logging.basicConfig(level=logging.DEBUG)


class OodCifar100(torchvision.datasets.CIFAR100):
    def __init__(self, data_path, transform, train, templates):
        super(OodCifar100, self).__init__(root=data_path,
                                          transform=transform,
                                          train=train,
                                          download=True)
        self.targets = np.array(self.targets)
        self.templates = templates if templates else cifar_templates



def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    cifar = OodCifar100(data_path, transform, train)
    get_distances_for_dataset(cifar, clip_model, "cifar100")


if __name__ == '__main__':
    main()

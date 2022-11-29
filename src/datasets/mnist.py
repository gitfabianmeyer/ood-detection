import logging

import clip
import torchvision.datasets

from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config

logging.basicConfig(level=logging.DEBUG)


class OodMNIST(torchvision.datasets.MNIST):
    def __init__(self, datapath, transform, train):
        super(OodMNIST, self).__init__(root=datapath,
                                       transform=transform,
                                       download=True,
                                       train=train)


class OodFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, datapath, transform, train):
        super(OodFashionMNIST, self).__init__(root=datapath,
                                              transform=transform,
                                              download=True,
                                              train=train)


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodMNIST(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "MNIST")

    dataset = OodFashionMNIST(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "Fashion MNIST")


if __name__ == '__main__':
    main()

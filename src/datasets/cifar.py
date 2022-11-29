import clip
import numpy as np
import torchvision.datasets
from ood_detection.config import Config

from metrics.distances import get_distances_for_dataset

from ood_detection.classification_utils import full_classification


class OodCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, data_path, transform, train):
        super(OodCifar10, self).__init__(root=data_path,
                                         transform=transform,
                                         train=train,
                                         download=True
                                         )
        self.targets = np.array(self.targets)


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodCifar10(data_path, transform, train)
    # get_distances_for_dataset(dataset, clip_model, "caltech101")
    full_classification(dataset, clip_model, "ood")
    dataset2 = torchvision.datasets.CIFAR10(data_path, transform=transform, train=train)
    full_classification(dataset, clip_model, "og")
    # get_distances_for_dataset(cifar, clip_model, "cifar10")


if __name__ == '__main__':
    main()

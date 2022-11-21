import clip
import numpy as np
import torchvision.datasets
from ood_detection.config import Config

from datasets.zoc_loader import single_isolated_class_loader

from metrics.distances import MaximumMeanDiscrepancy


class OodCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, datapath, transform, train):
        super(OodCifar10, self).__init__(root=datapath,
                                         transform=transform,
                                         train=train,
                                         download=True
                                         )
        self.targets = np.array(self.targets)


class OodCifar100(torchvision.datasets.CIFAR100):
    def __init__(self, datapath, transform, train):
        super(OodCifar100, self).__init__(root=datapath,
                                          transform=transform,
                                          train=train,
                                          download=True)
        self.targets = np.array(self.targets)


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)
    cifar = OodCifar10(data_path, transform, train)
    loaders = single_isolated_class_loader(cifar, batch_size=12)
    distancer = MaximumMeanDiscrepancy(loaders, cifar.classes, clip_model, )
    mean, std = distancer.get_distance_for_n_splits()
    print(f"Mean: {mean}, std: {std}")

if __name__ == '__main__':
    main()

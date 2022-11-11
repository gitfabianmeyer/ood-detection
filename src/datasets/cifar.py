import clip
import numpy as np
import torchvision.datasets
from ood_detection.config import Config

from src.datasets.zoc_loader import single_isolated_class_loader


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
    datapath = Config.DATAPATH
    train = False
    _, transform = clip.load(Config.VISION_MODEL)
    cifar = OodCifar100(datapath, transform, train)
    loaders = single_isolated_class_loader(cifar)

    for loader in loaders.keys():
        print(loader)
        for _ in loaders[loader]:
            print(10)


if __name__ == '__main__':
    main()

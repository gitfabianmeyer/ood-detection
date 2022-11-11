import clip
import torchvision.datasets

from src.datasets.zoc_loader import single_isolated_class_loader
from src.ood_detection.config import Config


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
    datapath = Config.DATAPATH
    train = False
    _, transform = clip.load(Config.VISION_MODEL)
    cifar = OodMNIST(datapath, transform, train)
    loaders = single_isolated_class_loader(cifar)

    for loader in loaders.keys():
        print(loader)
        for item in loaders[loader]:
            print(10)
            pass


if __name__ == '__main__':
    main()

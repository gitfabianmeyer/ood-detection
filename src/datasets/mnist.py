import clip
import torchvision.datasets

from datasets.zoc_loader import single_isolated_class_loader
from ood_detection.config import Config


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
    dataset = OodMNIST(datapath, transform, train)
    loaders = single_isolated_class_loader(dataset)

    for loader in loaders.keys():
        curr = loaders[loader]
        print(loader)
        for item in curr:
            print(item)
            pass


if __name__ == '__main__':
    main()

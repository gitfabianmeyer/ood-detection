import os.path

import clip
import torchvision.datasets

from src.datasets.zoc_loader import single_isolated_class_loader
from src.ood_detection.config import Config


class OodSVHN(torchvision.datasets.SVHN):
    def __init__(self, root, transform, train):
        self.root = os.path.join(root, 'svhn')
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        super(OodSVHN, self).__init__(root=self.root,
                                      transform=transform,
                                      download=True,
                                      split='train' if train else 'test')
        self.targets = self.labels
        self.classes = ['0 - zero', '1 - one', '2 - two', '3 - three',
                        '4 - four', '5 - five', '6 - six',
                        '7 - seven', '8 - eight', '9 - nine']
        self.class_to_idx = {self.classes[i]: i for i in range(10)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}


def main():
    datapath = Config.DATAPATH
    train = False
    _, transform = clip.load(Config.VISION_MODEL)
    dataset = OodSVHN(datapath, transform, train)
    loaders = single_isolated_class_loader(dataset)

    for loader in loaders.keys():
        print(loader)
        for item in loaders[loader]:
            print(10)
            pass


if __name__ == '__main__':
    main()

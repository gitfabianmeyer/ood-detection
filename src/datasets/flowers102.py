import clip
import numpy as np
import torch
import torchvision.datasets
from PIL import Image

from datasets.classnames import flowers_classes
from matplotlib import pyplot as plt

from datasets.zoc_loader import single_isolated_class_loader
from ood_detection.config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm


class OodFlowers102(torchvision.datasets.Flowers102):
    def __init__(self, datapath, transform, train):
        super().__init__(datapath,
                         transform=transform,
                         split='train' if train else 'val',
                         download=True)
        self.classes = flowers_classes
        self.data = self._image_files
        self.targets = np.array(self._labels)
        self.class_to_idx = {cls: i for (i, cls) in enumerate(self.classes)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}


def main():
    datapath = Config.DATAPATH
    train = False
    _, transform = clip.load(Config.VISION_MODEL)
    dataset = OodFlowers102(datapath, transform, train)
    loaders = single_isolated_class_loader(dataset)
    dataloader = DataLoader(dataset)

    for k, loader in enumerate(loaders.keys()):
        if k % 5 == 0:
            print(loader)
            for i, item in enumerate(loaders[loader]):
                if i % 35 == 0:
                    # check label in OG

                    found = False
                    for j, (im, targ) in enumerate(dataloader):
                        if torch.equal(im, item):
                            found = True
                            print(f"\nTargt: {targ}, should be: {dataset.class_to_idx[loader]}")

                    if not found:
                        print("somethings wrong with item")



if __name__ == '__main__':
    main()

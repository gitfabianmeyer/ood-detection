import logging

import clip
import numpy as np
import torchvision.datasets

from datasets.classnames import flowers_classes
from ood_detection.config import Config
from metrics.distances import get_distances_for_dataset

logging.basicConfig(level=logging.DEBUG)


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
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodFlowers102(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "Flowers102")


if __name__ == '__main__':
    main()

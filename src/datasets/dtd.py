import logging

import PIL
import clip
import numpy as np
import torchvision.datasets

from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config

logging.basicConfig(level=logging.DEBUG)

class OodDTD(torchvision.datasets.DTD):
    def __init__(self, datapath, preprocess, train):
        super().__init__(datapath,
                         transform=preprocess,
                         download=True,
                         split='train' if train else 'val')
        self.data = self._image_files
        self.targets = np.array(self._labels)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        image = PIL.Image.open(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodDTD(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "Describable Textures Dataset")


if __name__ == '__main__':
    main()

import logging

import PIL
import clip
import numpy as np
import torchvision.datasets
from datasets.classnames import dtd_templates

from metrics.distances import get_distances_for_dataset, run_full_distances
from ood_detection.config import Config

logging.basicConfig(level=logging.INFO)


class OodDTD(torchvision.datasets.DTD):
    def __init__(self, data_path, transform, train, templates=None):
        super().__init__(data_path,
                         transform=transform,
                         download=True,
                         split='train' if train else 'val')
        self.data = self._image_files
        self.targets = np.array(self._labels)
        self.templates = templates if templates else dtd_templates

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        image = PIL.Image.open(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def main():
    name = "dtd"
    dataset = OodDTD
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()


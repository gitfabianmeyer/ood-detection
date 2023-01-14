import logging

import PIL
import clip
import numpy as np
import torchvision
from datasets import corruptions
from metrics.distances import get_distances_for_dataset, run_full_distances
from ood_detection.config import Config
from datasets.classnames import stanfordcars_templates
from torchvision.transforms import Compose

logging.basicConfig(level=logging.INFO)


class OodStanfordCars(torchvision.datasets.StanfordCars):
    def __init__(self, data_path, transform, split, templates=None):
        super().__init__(data_path,
                         transform=transform,
                         download=True,
                         split='train' if split=='val' else split)
        self.data, self.targets = zip(*self._samples)
        self.targets = np.array(self.targets)
        self.templates = templates if templates else stanfordcars_templates

    def __getitem__(self, idx):

        image, label = self.data[idx], self.targets[idx]
        image = PIL.Image.open(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.data)


def main():
    name = "StanfordCars"
    dataset = OodStanfordCars
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()


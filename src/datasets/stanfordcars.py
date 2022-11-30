import logging

import PIL
import clip
import numpy as np
import torchvision
from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config
from datasets.classnames import stanfordcars_templates

logging.basicConfig(level=logging.INFO)


class OodStanfordCars(torchvision.datasets.StanfordCars):
    def __init__(self, data_path, transform, train, templates=None):
        super().__init__(data_path,
                         transform=transform,
                         download=True,
                         split='train' if train else 'test')
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
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodStanfordCars(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "StanfordCars")


if __name__ == '__main__':
    main()

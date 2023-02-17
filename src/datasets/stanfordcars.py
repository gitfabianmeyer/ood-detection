import logging

import PIL
import numpy as np
import torchvision
from ood_detection.config import Config
from datasets.classnames import stanfordcars_templates
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


class OodStanfordCars(torchvision.datasets.StanfordCars):
    def __init__(self, data_path, transform, split, templates=None):
        super().__init__(data_path,
                         transform=transform,
                         download=True,
                         split='train' if split == 'val' else split)
        self.split = split
        self.data, self.targets = zip(*self._samples)
        self.targets = np.array(self.targets)
        self.templates = templates if templates else stanfordcars_templates
        self.set_split()
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}
        self.name = 'stanford cars'

    def set_split(self):
        if self.split == 'val':
            _, self.data, _, self.targets = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)
        elif self.split == 'train':
            self.data, _, self.targets, _ = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)

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

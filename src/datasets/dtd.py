import logging

import PIL
import numpy as np
import torchvision.datasets
from datasets.classnames import dtd_templates

logging.basicConfig(level=logging.INFO)


class OodDTD(torchvision.datasets.DTD):
    def __init__(self, data_path, transform, split, templates=None):
        super().__init__(data_path,
                         transform=transform,
                         download=True,
                         split=split)
        self.data = self._image_files
        self.targets = np.array(self._labels)
        self.templates = templates if templates else dtd_templates
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}
        self.name = 'dtd'

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        image = PIL.Image.open(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.data)



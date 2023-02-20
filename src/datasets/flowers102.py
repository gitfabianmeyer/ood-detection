import logging
from typing import Tuple, Any

import PIL
import numpy as np
import torchvision
from datasets.classnames import flowers102_classes, flowers102_templates

logging.basicConfig(level=logging.INFO)


class OodFlowers102(torchvision.datasets.Flowers102):
    def __init__(self, data_path, transform, split, templates=None):
        super().__init__(data_path,
                         transform=transform,
                         split=split,
                         download=True)
        self.classes = flowers102_classes
        self.templates = templates if templates else flowers102_templates
        self.data = self._image_files
        self.targets = np.array(self._labels)
        self.class_to_idx = {cls: i for (i, cls) in enumerate(self.classes)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}
        self.name = 'flowers102'

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self.data[idx], self.targets[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

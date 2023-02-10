import logging
import os.path

import clip
import numpy as np
import torchvision.datasets
from PIL import Image
from datasets.classnames import caltech101_templates

from metrics.distances import run_full_distances
from ood_detection.config import Config
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


class OodCaltech101(torchvision.datasets.Caltech101):
    def __init__(self, data_path, transform, split, templates=None):
        super().__init__(data_path,
                         transform=transform,
                         download=True)
        self.targets = np.array(self.y)
        self.data = self.transform_to_image_list()
        self.classes = self.categories
        self.idx_to_class = {i: cls for (i, cls) in enumerate(self.classes)}
        self.class_to_idx = {value: key for (key, value) in self.idx_to_class.items()}
        self.templates = templates if templates else caltech101_templates
        self.data, self.targets = self.get_split(split)
        self.name = 'caltech101'

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.targets)

    def transform_to_image_list(self):
        # basically use the original __getitem__
        file_list = []
        for idx in range(len(self.targets)):
            path = os.path.join(self.root,
                                "101_ObjectCategories",
                                self.categories[self.targets[idx]],
                                f"image_{self.index[idx]:04d}.jpg")
            file_list.append(path)
        return file_list

    def get_split(self, split):

        x_train, x_test, y_train, y_test = train_test_split(self.data,
                                                            self.targets,
                                                            test_size=Config.TEST_SIZE,
                                                            stratify=self.targets,
                                                            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=Config.TEST_SIZE,
                                                          stratify=y_train,
                                                          random_state=42)
        if split == 'train':
            return x_train, y_train
        elif split == 'val':
            return x_val, y_val
        elif split == 'test':
            return x_test, y_test

        else:
            raise ValueError(f'Split {split} not in [train, test, val]')


def main():
    name = 'CALTECH101'
    dataset = OodCaltech101
    run_full_distances(name=name, dataset=dataset, lsun=False)


if __name__ == '__main__':
    main()

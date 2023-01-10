import logging
import os.path

import clip
import numpy as np
import torchvision.datasets
from PIL import Image
from datasets.classnames import caltech101_templates

from metrics.distances import run_full_distances
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


class OodCaltech101(torchvision.datasets.Caltech101):
    def __init__(self, data_path, transform, train, templates=None):
        super().__init__(data_path,
                         transform=transform,
                         # train=train,
                         download=True)
        self.targets = np.array(self.y)
        self.data = self.transform_to_image_list()
        self.classes = self.categories
        self.idx_to_class = {i: cls for (i, cls) in enumerate(self.classes)}
        self.class_to_idx = {value: key for (key, value) in self.idx_to_class.items()}
        self.templates = templates if templates else caltech101_templates
        self.data, self.targets = self.get_split(train)

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

    def get_split(self, train):

        x_train, x_test, y_train, y_test = train_test_split(self.data,
                                                            self.targets,
                                                            test_size=.3,
                                                            stratify=self.targets,
                                                            random_state=42)
        if train:
            return x_train, y_train
        else:
            return x_test, y_test


def main():
    name = 'CALTECH101'
    dataset = OodCaltech101
    run_full_distances(name=name, dataset=dataset, lsun=False)


if __name__ == '__main__':
    main()

import logging
import os.path

import clip
import torchvision.datasets

from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config
from datasets.classnames import mnist_templates

logging.basicConfig(level=logging.DEBUG)


class OodSVHN(torchvision.datasets.SVHN):
    def __init__(self, data_path, transform, train, templates=None):
        self.root = os.path.join(data_path, 'svhn')
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        super(OodSVHN, self).__init__(root=self.root,
                                      transform=transform,
                                      download=True,
                                      split='train' if train else 'test')
        self.targets = self.labels
        self.classes = [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
        ]
        self.class_to_idx = {self.classes[i]: i for i in range(10)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}
        self.templates = templates if templates else mnist_templates


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodSVHN(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "SVHN")


if __name__ == '__main__':
    main()

import logging

import clip
import torchvision.datasets
from datasets.classnames import mnist_templates

from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config

logging.basicConfig(level=logging.DEBUG)


class OodMNIST(torchvision.datasets.MNIST):
    def __init__(self, datapath, transform, train, templates=None):
        super(OodMNIST, self).__init__(root=datapath,
                                       transform=transform,
                                       download=True,
                                       train=train)

        self.templates = templates if templates else [
            'a photo of the number: "{}".',
        ]
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
        self.class_to_idx = {cls: i for (i, cls) in enumerate(self.classes)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodMNIST(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "MNIST")


if __name__ == '__main__':
    main()

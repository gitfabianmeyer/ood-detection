import os

from ood_detection.config import Config
from sklearn.model_selection import train_test_split

from datasets.mnist import CustomMNIST

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging

import torchvision

from datasets.classnames import mnist_templates
from metrics.distances import run_full_distances

logging.basicConfig(level=logging.INFO)


class CustomFashionMNIST(CustomMNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class OodFashionMNIST(CustomFashionMNIST):
    def __init__(self, data_path, transform, split, templates=None):
        super(OodFashionMNIST, self).__init__(root=data_path,
                                              transform=transform,
                                              download=True,
                                              train=True if split == 'train' or split == 'val' else False, )
        self.templates = templates if templates else mnist_templates
        self.split = split
        self.set_split()
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}
        self.name = 'fashion mnist'

    @property
    def class_to_idx(self):
        return super(OodFashionMNIST, self).class_to_idx

    @class_to_idx.setter
    def class_to_idx(self, value):
        self._class_to_idx = value

    def set_split(self):
        if self.split == 'val':
            _, self.data, _, self.targets = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)
        elif self.split == 'train':
            self.data, _, self.targets, _ = train_test_split(self.data, self.targets, test_size=Config.TEST_SIZE,
                                                             random_state=42, stratify=self.targets)


def main():
    dataset = OodFashionMNIST
    run_full_distances(dataset.name, dataset, lsun=False)


if __name__ == '__main__':
    main()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
import os.path

import torchvision.datasets


from metrics.distances import run_full_distances
from datasets.classnames import mnist_templates

logging.basicConfig(level=logging.INFO)


class OodSVHN(torchvision.datasets.SVHN):
    def __init__(self, data_path, transform, train, templates=None, classes=None):
        self.root = os.path.join(data_path, 'svhn')
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        super(OodSVHN, self).__init__(root=self.root,
                                      transform=transform,
                                      download=True,
                                      split='train' if train else 'test')
        self.targets = self.labels
        self.classes = classes if classes else ['0 - zero', '1 - one', '2 - two', '3 - three',
                                                '4 - four', '5 - five', '6 - six',
                                                '7 - seven', '8 - eight', '9 - nine']
        self.class_to_idx = {self.classes[i]: i for i in range(10)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}
        self.templates = templates if templates else mnist_templates


def main():
    name = "SVHN"
    dataset = OodSVHN
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()


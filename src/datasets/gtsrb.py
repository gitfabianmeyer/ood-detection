import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging

import clip
import numpy as np
import torchvision
from datasets.classnames import gtsrb_classes, gtsrb_templates
from ood_detection.config import Config

from metrics.distances import get_distances_for_dataset, run_full_distances

logging.basicConfig(level=logging.INFO)


class OodGTSRB(torchvision.datasets.GTSRB):
    def __init__(self, data_path, transform, split, templates=None):
        super().__init__(data_path,
                         split=split,
                         download=True,
                         transform=transform)
        self.classes = gtsrb_classes
        self.templates = templates if templates else gtsrb_templates
        self.class_to_idx = dict(zip(self.classes, list(range(len(self.classes)))))
        self.idx_to_class = dict(zip(list(range(len(self.classes))), self.classes))
        self.data, self.targets = zip(*self._samples)
        self.targets = np.array(self.targets)

    def __len__(self) -> int:
        return len(self.targets)


def main():
    name = "gtrsb"
    dataset = OodGTSRB
    run_full_distances(name, dataset, lsun=False)


if __name__ == '__main__':
    main()

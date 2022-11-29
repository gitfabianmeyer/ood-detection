import logging

import clip
import numpy as np
import torchvision
from datasets import classnames
from ood_detection.config import Config

from metrics.distances import get_distances_for_dataset

logging.basicConfig(level=logging.DEBUG)


class OodGTSRB(torchvision.datasets.GTSRB):
    def __init__(self, root, transform, train):
        super().__init__(root,
                         split="train" if train else "test",
                         download=True,
                         transform=transform)
        self.classes = classnames.gtsrb_classes
        self.class_to_idx = dict(zip(self.classes, list(range(len(self.classes)))))
        self.idx_to_class = dict(zip(list(range(len(self.classes))), self.classes))
        self.data, self.targets = zip(*self._samples)
        self.targets = np.array(self.targets)


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodGTSRB(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "GTRSB")


if __name__ == '__main__':
    main()

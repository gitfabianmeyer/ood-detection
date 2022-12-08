import os.path

import clip
import numpy as np
from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config
from torchvision.datasets import ImageFolder


class OodPneumonia(ImageFolder):
    def __init__(self, data_path, transform, train, templates=None):
        self.train = 'train' if train else 'val'
        self.root = os.path.join(data_path, 'chest_xray', self.train)
        super(OodPneumonia, self).__init__(self.root, transform)
        self.templates = templates if templates else "X-ray image of a chest with diagnosis "
        self.data, self.targets = zip(*self.samples)
        self.data = list(self.data)
        self.targets = np.array(self.targets)
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}


def main():
    data_path = Config.DATAPATH
    train = True
    clip_model, transform = clip.load(Config.VISION_MODEL)

    pneumonia = OodPneumonia(data_path, transform, train)
    get_distances_for_dataset(pneumonia, clip_model, "Pneumonia")


if __name__ == '__main__':
    main()

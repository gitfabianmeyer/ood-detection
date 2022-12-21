import logging
import clip
import numpy as np
import torchvision.datasets
from datasets.classnames import cifar_templates
from ood_detection.config import Config

from metrics.distances import get_distances_for_dataset, get_corruption_metrics

logging.basicConfig(level=logging.INFO)


class OodCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, data_path, transform, train, templates=None):
        super(OodCifar10, self).__init__(root=data_path,
                                         transform=transform,
                                         train=train,
                                         download=True
                                         )
        self.targets = np.array(self.targets)
        self.templates = templates if templates else cifar_templates
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform_clip = clip.load(Config.VISION_MODEL)
    get_corruption_metrics(OodCifar10, clip_model, transform_clip, "CIFAR10")

    cifar = OodCifar10(data_path, transform_clip, train)
    get_distances_for_dataset(cifar, clip_model, "cifar10")


if __name__ == '__main__':
    main()

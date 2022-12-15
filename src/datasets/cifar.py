import logging
import clip
import numpy as np
import torchvision.datasets
from datasets import corruptions
from datasets.classnames import cifar_templates
from ood_detection.config import Config

from metrics.distances import get_distances_for_dataset
from torchvision.transforms import Compose

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

    cifar = OodCifar10(data_path, transform_clip, train)
    get_distances_for_dataset(cifar, clip_model, "cifar10")

    corruption_dict = corruptions.Corruptions
    corr = "Gaussian Blur"
    for i in range(1, 6):
        print(f"Corruption {corr}, severity: {i}")
        corruption = corruption_dict[corr](severity=i)
        transform_list = transform_clip.transforms[:-1]
        transform_list.append(corruption)
        transform = Compose(transform_list)
        dataset = OodCifar10(data_path, transform, train)
        get_distances_for_dataset(dataset, clip_model, "CIFAR10", lsun=False, corruption=corr, severity=i)
        # full_batch_classification(dataset, clip_model, "LSUN")


if __name__ == '__main__':
    main()

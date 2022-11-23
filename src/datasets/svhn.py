import os.path

import clip
import torchvision.datasets

from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config


class OodSVHN(torchvision.datasets.SVHN):
    def __init__(self, root, transform, train):
        self.root = os.path.join(root, 'svhn')
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        super(OodSVHN, self).__init__(root=self.root,
                                      transform=transform,
                                      download=True,
                                      split='train' if train else 'test')
        self.targets = self.labels
        self.classes = ['0 - zero', '1 - one', '2 - two', '3 - three',
                        '4 - four', '5 - five', '6 - six',
                        '7 - seven', '8 - eight', '9 - nine']
        self.class_to_idx = {self.classes[i]: i for i in range(10)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodSVHN(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "MNIST")


if __name__ == '__main__':
    main()

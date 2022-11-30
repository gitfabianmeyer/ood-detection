import clip
import torchvision

from datasets.classnames import mnist_templates
from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config


class OodFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, data_path, transform, train, templates=None):
        super(OodFashionMNIST, self).__init__(root=data_path,
                                              transform=transform,
                                              download=True,
                                              train=train)
        self.templates = templates if templates else mnist_templates


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodFashionMNIST(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "MNIST")


if __name__ == '__main__':
    main()

import clip
import torchvision.datasets

from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config


class OodDTD(torchvision.datasets.DTD):
    def __init__(self, datapath, preprocess, train):
        super().__init__(datapath,
                         transform=preprocess,
                         download=True,
                         split='train' if train else 'val')
        self._images = self._image_files


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodDTD(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "Describable Textures Dataset")


if __name__ == '__main__':
    main()

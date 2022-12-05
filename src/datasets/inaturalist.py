import os.path

import clip
import torchvision.datasets
from ood_detection.classification_utils import full_batch_classification
from ood_detection.config import Config


class OodINaturalist(torchvision.datasets.INaturalist):
    def __init__(self, data_path, transform, train):
        version = "2021_train" if train else "2021_valid"
        self.root = os.path.join(data_path, "iNaturalist")
        download = False if os.path.exists(os.path.join(self.root, version)) else True
        super(OodINaturalist, self).__init__(root=self.root,
                                             transform=transform,
                                             download=download,
                                             version=version
                                             )


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodINaturalist(data_path, transform, train)
    full_batch_classification(dataset, clip_model, "iNaturalist")
    # get_distances_for_dataset(dataset, clip_model, "GTRSB")


if __name__ == '__main__':
    main()


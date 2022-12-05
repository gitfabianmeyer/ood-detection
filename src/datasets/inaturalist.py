import clip
import torchvision.datasets
from ood_detection.classification_utils import full_batch_classification
from ood_detection.config import Config


class OodINaturalist(torchvision.datasets.INaturalist):
    def __init__(self, data_path, transform, train):
        super(OodINaturalist, self).__init__(root=data_path,
                                             transform=transform,
                                             download=True,
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


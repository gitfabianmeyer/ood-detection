import clip
import numpy as np
import torchvision
from PIL import Image
from ood_detection.config import Config
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from metrics.distances import get_distances_for_dataset


class OodGTSRB(torchvision.datasets.GTSRB):
    def __init__(self, root, transform, train):
        super().__init__(root,
                         split="train" if train else "test",
                         download=True,
                         transform=transform)
        self.classes = [
            'red and white circle 20 kph speed limit',
            'red and white circle 30 kph speed limit',
            'red and white circle 50 kph speed limit',
            'red and white circle 60 kph speed limit',
            'red and white circle 70 kph speed limit',
            'red and white circle 80 kph speed limit',
            'end / de-restriction of 80 kph speed limit',
            'red and white circle 100 kph speed limit',
            'red and white circle 120 kph speed limit',
            'red and white circle red car and black car no passing',
            'red and white circle red truck and black car no passing',
            'red and white triangle road intersection warning',
            'white and yellow diamond priority road',
            'red and white upside down triangle yield right-of-way',
            'stop',
            'empty red and white circle',
            'red and white circle no truck entry',
            'red circle with white horizonal stripe no entry',
            'red and white triangle with exclamation mark warning',
            'red and white triangle with black left curve approaching warning',
            'red and white triangle with black right curve approaching warning',
            'red and white triangle with black double curve approaching warning',
            'red and white triangle rough / bumpy road warning',
            'red and white triangle car skidding / slipping warning',
            'red and white triangle with merging / narrow lanes warning',
            'red and white triangle with person digging / construction / road work warning',
            'red and white triangle with traffic light approaching warning',
            'red and white triangle with person walking warning',
            'red and white triangle with child and person walking warning',
            'red and white triangle with bicyle warning',
            'red and white triangle with snowflake / ice warning',
            'red and white triangle with deer warning',
            'white circle with gray strike bar no speed limit',
            'blue circle with white right turn arrow mandatory',
            'blue circle with white left turn arrow mandatory',
            'blue circle with white forward arrow mandatory',
            'blue circle with white forward or right turn arrow mandatory',
            'blue circle with white forward or left turn arrow mandatory',
            'blue circle with white keep right arrow mandatory',
            'blue circle with white keep left arrow mandatory',
            'blue circle with white arrows indicating a traffic circle',
            'white circle with gray strike bar indicating no passing for cars has ended',
            'white circle with gray strike bar indicating no passing for trucks has ended',
        ]
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

import clip
import numpy as np
import torch
import torchvision.datasets

from datasets.classnames import flowers_classes
from ood_detection.config import Config
from metrics.distances import get_distances_for_dataset


class OodFlowers102(torchvision.datasets.Flowers102):
    def __init__(self, datapath, transform, train):
        super().__init__(datapath,
                         transform=transform,
                         split='train' if train else 'val',
                         download=True)
        self.classes = flowers_classes
        self.data = self._image_files
        self.targets = np.array(self._labels)
        self.class_to_idx = {cls: i for (i, cls) in enumerate(self.classes)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodFlowers102(data_path, transform, train)
    # get_distances_for_dataset(dataset, clip_model, "caltech101")
    dataset2 = torchvision.datasets.Flowers102(data_path, transform=transform, split='val')

    dataloader1 = torch.utils.data.DataLoader(dataset, batch_size=256)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=256)

    i = 0
    j = 0
    for (img1, targ1), (img2, targ2) in zip(dataloader1, dataloader2):
        j += 1
        if not torch.equal(targ1, targ2):
            print(targ1)
            print(targ2)

        if not torch.equal(img1, img2):
            i += 1
            # print("unequal images")

        print(f"{i} / {j}")


if __name__ == '__main__':
    main()

import os.path

import clip
import numpy as np
import torch.utils.data
import torchvision.datasets
from PIL import Image
from ood_detection.config import Config

from metrics.distances import get_distances_for_dataset


class OodCaltech101(torchvision.datasets.Caltech101):
    def __init__(self, datapath, transform, train):
        super().__init__(datapath,
                         transform=transform,
                         # train=train,
                         download=True)
        self.targets = np.array(self.y)
        self.data = self.transform_to_image_list()
        self.classes = self.categories
        self.idx_to_class = {i: cls for (i, cls) in enumerate(self.classes)}
        self.class_to_idx = {value: key for (key, value) in self.idx_to_class.items()}

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def transform_to_image_list(self):
        # basically use the original __getitem__
        file_list = []
        for i in range(len(self.index)):
            path = os.path.join(self.root,
                                "101_ObjectCategories",
                                self.categories[self.targets[i]],
                                f"image_{self.index[i]:04d}.jpg")
            file_list.append(path)
        return file_list


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodCaltech101(data_path, transform, train)
    # get_distances_for_dataset(dataset, clip_model, "caltech101")
    dataset2 = torchvision.datasets.Caltech101(data_path, transform=transform)

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

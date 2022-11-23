import os.path

import clip
import torchvision.datasets
from PIL import Image
from ood_detection.config import Config
from metrics.distances import get_distances_for_dataset


class OodCaltech101(torchvision.datasets.Caltech101):
    def __init__(self, datapath, transform, train):
        super().__init__(datapath,
                         transform=transform,
                         train=train,
                         download=True)
        self._labels = self.y
        self._images = self.transform_to_image_list()

    def __getitem__(self, idx):
        img = Image.open(self._images[idx])
        if self.transform is not None:
            img = self.transform(img)
        target = self._labels[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def transform_to_image_list(self):
        # basically use the original __getitem__
        file_list = []
        for i in self.index:
            path = os.path.join(self.root,
                                "101_ObjectCategories",
                                self.categories[self._labels[i]],
                                f"image_{self.index[i]:04d}.jpg")
            file_list.append(path)
        return file_list


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodCaltech101(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "caltech101")


if __name__ == '__main__':
    main()

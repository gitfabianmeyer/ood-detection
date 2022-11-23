import PIL
import clip
import torchvision
from datasets.svhn import OodSVHN
from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config


class OodStanfordCars(torchvision.datasets.StanfordCars):
    def __init__(self, datapath, transform):
        super().__init__(datapath,
                         transform=transform,
                         download=True)
        self._labels = [tup[1] for tup in self._samples]
        self._images = [tup[0] for tup in self._samples]

    def __getitem__(self, idx):

        image_file, label = self._images[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self._labels)


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodSVHN(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "MNIST")


if __name__ == '__main__':
    main()

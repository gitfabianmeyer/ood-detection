import PIL
import torchvision.datasets


class StandardizedStanfordCars(torchvision.datasets.StanfordCars):
    def __init__(self, datapath, transform):
        super().__init__(datapath,
                         transform=transform,
                         download=True)
        self._labels = [tup[1] for tup in self._samples]
        self._image_files = [tup[0] for tup in self._samples]

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

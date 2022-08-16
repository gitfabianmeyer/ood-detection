import torchvision.datasets

from ood_detection.classnames import flowers_classes


class FlowersWithLabels(torchvision.datasets.Flowers102):
    def __init__(self, datapath, transform):
        super().__init__(datapath,
                         transform=transform,
                         download=True)
        self.classes = flowers_classes
        self._images = self.image_files
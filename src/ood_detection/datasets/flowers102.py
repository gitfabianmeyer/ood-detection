import torchvision.datasets

from ood_detection.classnames import flowers_classes


class FlowersWithLabels(torchvision.datasets.Flowers102):
    def __init__(self):
        super.__init__()
        self.classes = flowers_classes
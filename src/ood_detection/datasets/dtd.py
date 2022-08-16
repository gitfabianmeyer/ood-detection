import torchvision.datasets


class StandardizedDTD(torchvision.datasets.DTD):
    def __init__(self, datapath, preprocess):
        super().__init__(datapath, transform=preprocess)
        self._images = self._image_files


import PIL.Image
import numpy as np
import torchvision.datasets
from PIL import Image
from ood_detection import classnames
from ood_detection.config import Config
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize


class aircraft_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(aircraft_isolated_class, self).__init__()
        self.transform = Compose([
            # ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        aircraft = torchvision.datasets.FGVCAircraft(root=Config.DATAPATH, split='val', download=True)
        class_mask = np.array(aircraft._labels) == aircraft.class_to_idx[class_label]
        self.data = [aircraft._image_files[i] for i in range(len(aircraft._image_files)) if class_mask[i]]
        self.targets = np.array(aircraft._labels)[class_mask]

    def __getitem__(self, idx):
        image_file = self.data[idx]
        image = PIL.Image.open(image_file).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.data)


def aircraft_single_isolated_class_loader(batch_size=1):
    loaders_dict = {}
    labels = classnames.fgvcaircraft_classes
    for label in labels:
        dataset = aircraft_isolated_class(label)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
        loaders_dict[label] = loader

    return loaders_dict


def get_aircraft_loader(transform):
    aircraft = torchvision.datasets.FGVCAircraft(root=Config.DATAPATH, transform=transform,
                                                 split='val', download=True)
    loader = DataLoader(dataset=aircraft, batch_size=128)
    return loader

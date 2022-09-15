import numpy as np
import torchvision.datasets.vision
from PIL import Image
from ood_detection import classnames
from ood_detection.config import Config
from ood_detection.datasets.stanfordcars import StandardizedStanfordCars
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize


def get_cars_loader(transform):
    cars = torchvision.datasets.StanfordCars(root=Config.DATAPATH, transform=transform,
                                             split='test', download=True)
    loader = DataLoader(dataset=cars, batch_size=64)
    return loader


class cars_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(cars_isolated_class, self).__init__()
        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        cars = torchvision.datasets.StanfordCars(root='./data', split='test', download=True)

        class_mask = np.array(cars.targets) == cars.class_to_idx[class_label]
        self.data = cars.data[class_mask]
        self.targets = np.array(cars.targets)[class_mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])


def cars_single_isolated_class_loader():
    loaders_dict = {}
    cars_labels = classnames.stanfordcars_classes
    for label in cars_labels:
        dataset = cars_isolated_class(label)
        loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
        loaders_dict[label] = loader
    return loaders_dict

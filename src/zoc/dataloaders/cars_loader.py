import PIL
import numpy as np
import torchvision.datasets.vision
from PIL import Image
from ood_detection import classnames
from ood_detection.config import Config
from torch.utils.data import Dataset, DataLoader
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
            # ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        cars = torchvision.datasets.StanfordCars(root=Config.DATAPATH, split='test', download=True)
        labels = [sample[1] for sample in cars._samples]
        class_label_id = cars.class_to_idx[class_label]
        class_mask = np.array(labels) == class_label_id
        tuples = [cars._samples[i] for i in range(len(cars._samples)) if class_mask[i]]
        self.data, self.targets = zip(*tuples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_file = self.data[idx]
        image = PIL.Image.open(image_file).convert('RGB')
        return self.transform(image)


def cars_single_isolated_class_loader(batch_size=1):
    loaders_dict = {}
    cars_labels = torchvision.datasets.StanfordCars(root=Config.DATAPATH, split='test', download=True).class_to_idx.keys()
    for label in cars_labels:
        dataset = cars_isolated_class(label)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
        loaders_dict[label] = loader
    return loaders_dict

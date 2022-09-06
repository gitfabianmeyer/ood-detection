import PIL.Image
import numpy as np
from PIL import Image
from ood_detection import classnames
from ood_detection.config import Config
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DTD, OxfordIIITPet
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize


class oxford3_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(oxford3_isolated_class, self).__init__()
        self.transform = Compose([
            # ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        oxford3pets = OxfordIIITPet(root=Config.DATAPATH, split='test', download=True)
        class_mask = np.array(oxford3pets._labels) == oxford3pets.class_to_idx[class_label]
        self.data = [oxford3pets._image_files[i] for i in range(len(oxford3pets._image_files)) if class_mask[i]]
        self.targets = np.array(oxford3pets._labels)[class_mask]

    def __getitem__(self, idx):
        image_file = self.data[idx]
        image = PIL.Image.open(image_file).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.data)


def oxford3_single_isolated_class_loader():
    loaders_dict = {}
    labels = classnames.dtd_classes
    for label in labels:
        dataset = oxford3_isolated_class(label)
        loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
        loaders_dict[label] = loader

    return loaders_dict


def get_oxfordiiipets_loader(transform):
    dtd = OxfordIIITPet(root=Config.DATAPATH, transform=transform,
              split='test', download=True)
    loader = DataLoader(dataset=dtd, batch_size=64)
    return loader

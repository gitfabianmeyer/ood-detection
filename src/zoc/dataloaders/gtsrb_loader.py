import PIL.Image
import numpy as np
import torchvision.datasets
from PIL import Image
from ood_detection import classnames
from ood_detection.config import Config
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize


class gtrsb_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label or class_label == 0, 'a semantic label should be specified'
        super(gtrsb_isolated_class, self).__init__()
        self.transform = Compose([
            # ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        gtrsb = torchvision.datasets.GTSRB(root=Config.DATAPATH, split='test', download=True)
        labels = [sample[1] for sample in gtrsb._samples]
        class_mask = np.array(labels) == class_label
        tuples = [gtrsb._samples[i] for i in range(len(gtrsb._samples)) if class_mask[i]]
        self.data, self.targets = zip(*tuples)

    def __getitem__(self, idx):
        image_file = self.data[idx]
        image = PIL.Image.open(image_file).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.data)


def gtrsb_single_isolated_class_loader():
    loaders_dict = {}
    labels = classnames.gtrsb_classes
    for label in labels:
        dataset = gtrsb_isolated_class(label)
        loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
        loaders_dict[label] = loader

    return loaders_dict


def get_gtrsb_loader(transform):
    gtrsb = torchvision.datasets.GTSRB(root=Config.DATAPATH, transform=transform,
                                       split='test', download=True)
    loader = DataLoader(dataset=gtrsb, batch_size=128)
    return loader

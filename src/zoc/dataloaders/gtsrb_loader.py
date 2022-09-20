import PIL.Image
import numpy as np
import torchvision.datasets
from PIL import Image
from ood_detection import classnames
from ood_detection.config import Config
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize

from ood_detection.datasets.gtsrb import StandardizedGTSRB


class gtsrb_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(gtsrb_isolated_class, self).__init__()
        self.transform = Compose([
            # ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        gtsrb = StandardizedGTSRB(root=Config.DATAPATH, split='test')
        labels = [sample[1] for sample in gtsrb._samples]
        class_mask = np.array(labels) == class_label
        self.data = [self.data[i] for i in class_mask if i]
        tuples = [gtsrb._samples[i] for i in range(len(gtsrb._samples)) if class_mask[i]]
        self.data, self.targets = zip(*tuples)

    def __getitem__(self, idx):
        image_file = self.data[idx]
        image = PIL.Image.open(image_file).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.data)


def gtsrb_single_isolated_class_loader(batch_size=1):
    loaders_dict = {}
    labels = classnames.gtsrb_classes
    for label in labels:
        dataset = gtsrb_isolated_class(label)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=1)
        loaders_dict[label] = loader

    return loaders_dict


def get_gtsrb_loader(transform):
    gtsrb = StandardizedGTSRB(root=Config.DATAPATH, transform=transform,
                              split='test')
    loader = DataLoader(dataset=gtsrb, batch_size=128)
    return loader

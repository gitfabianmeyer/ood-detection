import numpy as np
from PIL import Image
from ood_detection.config import Config
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize


def get_cifar10_loader(transform):
    cifar10 = CIFAR10(root=Config.DATAPATH, transform=transform,
                      train=False, download=True)
    loader = DataLoader(dataset=cifar10, batch_size=64)
    return loader


class cifar10_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(cifar10_isolated_class, self).__init__()
        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        cifar10 = CIFAR10(root=Config.DATAPATH, train=False, download=True)

        class_mask = np.array(cifar10.targets) == cifar10.class_to_idx[class_label]
        self.data = cifar10.data[class_mask]
        self.targets = np.array(cifar10.targets)[class_mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])


def cifar10_single_isolated_class_loader(batch_size=1):
    loaders_dict = {}
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for label in cifar10_labels:
        dataset = cifar10_isolated_class(label)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
        loaders_dict[label] = loader
    return loaders_dict


import PIL.Image
import numpy as np
from PIL import Image
from ood_detection import classnames
from ood_detection.config import Config
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DTD
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize


class dtd_isolated_class(Dataset):
    def __init__(self, class_label=None):
        assert class_label, 'a semantic label should be specified'
        super(dtd_isolated_class, self).__init__()
        self.transform = Compose([
            # ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        dtd = DTD(root=Config.DATAPATH, split='val', download=True)
        class_mask = np.array(dtd._labels) == dtd.class_to_idx[class_label]
        self.data = [dtd._image_files[i] for i in range(len(dtd._image_files)) if class_mask[i]]
        self.targets = np.array(dtd._labels)[class_mask]

    def __getitem__(self, idx):
        image_file = self.data[idx]
        image = PIL.Image.open(image_file).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.data)


def dtd_single_isolated_class_loader():
    loaders_dict = {}
    labels = classnames.dtd_classes
    for label in labels:
        dataset = dtd_isolated_class(label)
        loader = DataLoader(dataset=dataset, batch_size=1, num_workers=4)
        loaders_dict[label] = loader

    return loaders_dict

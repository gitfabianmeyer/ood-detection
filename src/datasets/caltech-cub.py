import os

import clip
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, DataLoader

from src.datasets.zoc_loader import single_isolated_class_loader
from src.ood_detection.config import Config


class OodCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, transform=None, train=True, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = []
        labels = []
        is_training = []
        path = os.path.join(self.root, self.base_folder)
        with open(os.path.join(self.root, 'CUB_200_2011', 'images.txt')) as f:
            for line in f:
                curr_img = line.rstrip("\n").split(" ")
                images.append(os.path.join(path,curr_img[1]))
        with open(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt')) as f:
            for line in f:
                curr_lab = line.rstrip("\n").split(" ")
                labels.append(int(curr_lab[1]))
        with open(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')) as f:
            for line in f:
                curr_split = line.rstrip("\n").split(" ")
                is_training.append(int(curr_split[1]))

        assert len(images) == len(labels), f'Something went wrong: {len(images)} to {len(labels)}'

        self.classes = []
        with open(os.path.join(self.root, 'CUB_200_2011', 'classes.txt')) as f:
            for line in f:
                curr_lab = line.rstrip("\n").split(" ")[1]
                self.classes.append(curr_lab)
        self.class_to_idx = {cls: i + 1 for (i, cls) in enumerate(self.classes)}
        self.idx_to_classes = {value: key for (key, value) in self.class_to_idx.items()}

        self.data = np.array(images)
        self.targets = np.array(labels)
        is_training = np.array(is_training, dtype=bool)

        if self.train:
            self.data = self.data[is_training]
            self.targets = self.targets[is_training]
        else:
            is_val = np.invert(is_training)
            self.data = self.data[is_val]
            self.targets = self.targets[is_val]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, image in enumerate(self.data):
            filepath = os.path.join(self.root, self.base_folder, image)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def main():
    datapath = Config.DATAPATH
    train = False
    _, transform = clip.load(Config.VISION_MODEL)
    dataset = OodCub2011(datapath, transform, train)

    dataloader = DataLoader(dataset)
    for i, (img, lab) in enumerate(dataloader):
        print(lab)
        if i % 10 == 0 and i != 0:
            break
    loaders = single_isolated_class_loader(dataset)

    for loader in loaders.keys():
        print(loader)
        dataloader = loaders[loader]
        for i, item in enumerate(dataloader):
            if i % 100 and i != 0:
                print(item)
                break


if __name__ == '__main__':
    main()

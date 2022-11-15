import os

import clip
import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from src.datasets.zoc_loader import single_isolated_class_loader
from src.ood_detection.config import Config


class OodCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
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

        with open(os.path.join(self.root, 'CUB_200_2011', 'images.txt')) as f:
            for line in f:
                curr_img = line.rstrip("\n").split(" ")
                images.append(curr_img[1])
        with open(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt')) as f:
            for line in f:
                curr_lab = line.rstrip("\n").split(" ")
                labels.append(int(curr_lab[1]))
        with open(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')) as f:
            for line in f:
                curr_split = line.rstrip("\n").split(" ")
                is_training.append(curr_split[1])

        assert len(images) == len(labels), f'Something went wrong: {len(images)} to {len(labels)}'

        self.classes = []
        with open(os.path.join(self.root, 'CUB_200_2011', 'classes.txt')) as f:
            for line in f:
                curr_lab = line.rstrip("\n").split(" ")[1]
                self.classes.append(curr_lab)
        self.class_to_idx = {cls: i + 1 for (i, cls) in enumerate(self.classes)}
        self.idx_to_classes = {value: key for (key, value) in self.class_to_idx.items()}

        if self.train:
            self.data = [(images[i], labels[i]) for i in range(len(images)) if is_training[i]]
        else:
            self.data = [(images[i], labels[i]) for i in range(len(images)) if not is_training[i]]

        self.data, self.targets = zip(*self.data)
        self.data = list(self.data)
        self.targets = np.array(self.targets)

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
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def main():
    datapath = Config.DATAPATH
    train = False
    _, transform = clip.load(Config.VISION_MODEL)
    dataset = OodCub2011(datapath, transform, train)
    loaders = single_isolated_class_loader(dataset)

    for loader in loaders.keys():
        print(loader)
        for item in loaders[loader]:
            print(10)
            pass


if __name__ == '__main__':
    main()

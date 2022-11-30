import logging
import os

import clip
import numpy as np
from PIL import Image
from datasets.classnames import imagenet_templates
from ood_detection.classification_utils import full_classification
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from ood_detection.config import Config
from metrics.distances import get_distances_for_dataset

logging.basicConfig(level=logging.INFO)


class OodCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, transform=None, train=True, download=True, templates=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.templates = templates if templates else imagenet_templates

    def _load_metadata(self):
        images = []
        labels = []
        is_training = []
        path = os.path.join(self.root, self.base_folder)
        with open(os.path.join(self.root, 'CUB_200_2011', 'images.txt')) as f:
            for line in f:
                curr_img = line.rstrip("\n").split(" ")
                images.append(os.path.join(path, curr_img[1]))
        with open(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt')) as f:
            for line in f:
                curr_class = line.rstrip("\n").split(" ")
                labels.append(int(curr_class[1]))
        with open(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')) as f:
            for line in f:
                curr_split = line.rstrip("\n").split(" ")
                is_training.append(int(curr_split[1]))

        assert len(images) == len(labels), f'Something went wrong: {len(images)} to {len(labels)}'

        self.classes = []
        with open(os.path.join(self.root, 'CUB_200_2011', 'classes.txt')) as f:
            for line in f:
                curr_class = line.rstrip("\n").split(" ")[1]
                curr_class = curr_class.split(".")[1]
                curr_class = " ".join(curr_class.split("_"))
                self.classes.append(curr_class)
        self.class_to_idx = {cls: i + 1 for (i, cls) in enumerate(self.classes)}
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}

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
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodCub2011(data_path, transform, train)
    full_classification(dataset, clip_model, "imagenet templates")
    templates = ["An close-up image of {}, a type of bird",]
    dataset = OodCub2011(data_path, transform, train, templates=templates)
    full_classification(dataset, clip_model, "template 1")
    templates = ["An close-up image of {}, a type of bird",
                 "A hobby photography of the bird {}.",
                 "{} from the species bird.",
                 "This is the {}, a bird."]
    dataset = OodCub2011(data_path, transform, train, templates=templates)
    full_classification(dataset, clip_model, "template 2")
    # get_distances_for_dataset(dataset, clip_model, "CUB2011")


if __name__ == '__main__':
    main()

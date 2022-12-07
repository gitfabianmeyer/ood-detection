import logging
import os

import clip
import numpy as np
import pandas as pd
from PIL import Image
from datasets.dataset_utils import download_and_extract
from ood_detection.classification_utils import full_batch_classification
from ood_detection.config import Config
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)


class OodCXRDataset(Dataset):

    def __init__(self, data_path, train, transform, templates=None):

        self.train = "train" if train else "val"
        self.root = os.path.join(data_path, "chestxrays")
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.image_dir = os.path.join(self.root, 'images')
        self._download()
        self.transform = transform
        self.index_dir = os.path.join(self.root, self.train + '_label.csv')
        self.classes = pd.read_csv(self.index_dir, header=None, nrows=1).ix[0, :].as_matrix()[1:9]
        self.label_index = pd.read_csv(self.index_dir, header=0)

        self.templates = templates
        self.class_to_idx = {
            'Atelectasis': 0,
            'Cardiomegaly': 1,
            'Effusion': 2,
            'Infiltrate': 3,
            'Mass': 4,
            'Nodule': 5,
            'Pneumonia': 6,
            'Pneumothorax': 7,
            'Consolidation': 8,
            'Edema': 9,
            'Emphysema': 10,
            'Fibrosis': 11,
            'Pleural_Thickening': 12,
            'Hernia': 13,
        }
        self.idx_to_class = {value: key for (key, value) in self.class_to_idx.items()}

    def _download(self):
        if os.path.exists(self.image_dir):
            return True
        links = [
            'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
            'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
            'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
            'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
            'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
            'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
            'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
            'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
            'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
            'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
            'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
            'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
        ]
        os.makedirs(self.image_dir, exist_ok=True)
        for link in links:
            download_and_extract(url=link,
                                 dirname=self.root)

    def __len__(self):
        return int(len(self.label_index) * 0.1)

    def __getitem__(self, idx):
        name = self.label_index.iloc[idx, 0]
        img_dir = os.path.join(self.image_dir, name)
        image = Image.open(img_dir).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.label_index.iloc[idx, 1:9].as_matrix().astype('int')

        return image, label


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodCXRDataset(data_path, transform, train)
    full_batch_classification(dataset, clip_model, "iNaturalist")
    # get_distances_for_dataset(dataset, clip_model, "GTRSB")


if __name__ == '__main__':
    main()

import logging
import shlex

import clip
import imageio
import numpy as np
import os
import subprocess

from datasets.classnames import imagenet_templates
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from tqdm.autonotebook import tqdm

from ood_detection.config import Config
from metrics.distances import get_distances_for_dataset

logging.basicConfig(level=logging.DEBUG)


def download_and_unzip(root_dir):
    shell_dir = os.path.dirname(os.path.dirname(os.path.normpath(root_dir)))
    shell_path = os.path.join(shell_dir, 'shell')
    shell_path = os.path.join(shell_path, 'tinyimagenet.sh')
    subprocess.call(shlex.split(f' {shell_path} {root_dir}'))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


"""Creates a paths datastructure for the tiny imagenet.
Args:
  root_dir: Where the data is located
  download: Download if the data is not there
Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:
"""


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            if os.path.exists(root_dir):
                print("TinyImagenet Folder already exists")
            else:
                download_and_unzip(os.path.dirname(root_dir))
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = {}
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                label = list(map(lambda x: x.strip(), labels.split(',')))[0]
                if nid in self.ids:
                    self.nid_to_words[nid] = label

        # num labels
        self.ids_to_num = {}
        self.num_to_ids = {}
        for i, idx in enumerate(self.ids):
            self.ids_to_num[idx] = i
            self.num_to_ids[i] = idx

        self.paths = {'train': [], 'val': [], 'test': list(map(lambda x: os.path.join(test_path, x),
                                                               os.listdir(test_path)))}

        # Get the test paths
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, _, _, _, _ = line.split()
                fname = os.path.join(val_path, nid, fname)
                self.paths['val'].append((fname, self.ids_to_num[nid]))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            # anno_path = os.path.join(train_path, nid, nid + '_boxes.txt')
            imgs_path = os.path.join(train_path, nid)
            label_id = self.ids_to_num[nid]
            for img_path in os.listdir(imgs_path):
                fname = os.path.join(imgs_path, img_path)
                self.paths['train'].append((fname, label_id))

            # with open(anno_path, 'r') as annof:
            #    for line in annof:
            #        fname, x0, y0, x1, y1 = line.split()
            #        fname = os.path.join(imgs_path, fname)
            #        bbox = int(x0), int(y0), int(x1), int(y1)
            #        self.paths['train'].append((fname, label_id, nid, bbox))


"""Datastructure for the tiny image dataset.
Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset
Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', preload=False, load_transform=None,
                 transform=None, download=False, max_samples=None):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.data = []
        self.targets = []
        self.classes = list(tinp.nid_to_words.values())
        self.class_to_idx = {value: key for key, value in tinp.nid_to_words.items()}
        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                 dtype=np.float32)
            self.targets = np.zeros((self.samples_num,), dtype=np.int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.data[idx] = img
                if mode != 'test':
                    self.targets[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.data, self.targets)
                    self.data, self.targets = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.data[idx]
            lbl = None if self.mode == 'test' else self.targets[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            img = _add_channels(img)
            lbl = None if self.mode == 'test' else s[self.label_idx]
        sample = {'image': img, 'label': lbl}

        if self.transform:
            sample = self.transform(sample)
        return sample


class TinyImageNetImageFolder(ImageFolder):
    def __init__(self, root, transform, train, download=True):

        self.words_to_nid = {}
        self.nid_to_words = {}
        self.dataset_root = root
        if download:
            if os.path.exists(self.dataset_root):
                print("TinyImagenet Folder already exists")
            else:
                download_and_unzip(os.path.dirname(self.dataset_root))

        if train:
            super(TinyImageNetImageFolder, self).__init__(os.path.join(self.dataset_root, 'train'), transform)
        else:
            super(TinyImageNetImageFolder, self).__init__(os.path.join(self.dataset_root, 'val'))

        self.data, _ = zip(*self.samples)
        self.targets = np.array(self.targets)
        self.init_semantic_labels()

    def init_semantic_labels(self):
        with open(os.path.join(self.dataset_root, 'words.txt'), 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                label = list(map(lambda x: x.strip(), labels.split(',')))[0]
                if nid in self.classes:
                    self.nid_to_words[nid] = label

        self.words_to_nid = {value: key for (key, value) in self.nid_to_words.items()}
        self.classes = list(self.words_to_nid.keys())
        self.class_to_idx = {self.nid_to_words[key]: value for (key, value) in self.class_to_idx.items()}


class OodTinyImageNet(TinyImageNetImageFolder):
    def __init__(self, data_path, transform, train, templates=None):
        super(OodTinyImageNet, self).__init__(root=os.path.join(data_path, 'tinyimagenet/tiny-imagenet-200'),
                                              transform=transform,
                                              train=train,
                                              download=True
                                              )
        self.transform = transform
        self.templates = templates if templates else imagenet_templates


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodTinyImageNet(data_path, transform, train)
    get_distances_for_dataset(dataset, clip_model, "TinyImagenet")


if __name__ == '__main__':
    main()

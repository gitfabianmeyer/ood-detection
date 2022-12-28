import io
import os
import logging
import pickle
import string
from typing import Any, Callable, cast, List, Optional, Tuple, Union, Iterable

import clip
import numpy as np
import torchvision
import subprocess
from urllib.request import Request, urlopen

from PIL import Image
from datasets import corruptions
from datasets.classnames import imagenet_templates
from metrics.distances import get_distances_for_dataset, get_corruption_metrics, run_full_distances
from ood_detection.classification_utils import full_batch_classification
from ood_detection.config import Config
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, iterable_to_str
from torchvision.transforms import Compose
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def list_categories():
    url = 'http://dl.yf.io/lsun/categories.txt'
    with urlopen(Request(url)) as response:
        return response.read().decode().strip().split('\n')


def download(out_dir, category, set_name):
    url = 'http://dl.yf.io/lsun/scenes/{category}_' \
          '{set_name}_lmdb.zip'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
        url = 'http://dl.yf.io/lsun/scenes/{set_name}_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    _logger.info('Downloading', category, set_name, 'set')
    subprocess.call(cmd)


class LSUNClass(VisionDataset):
    def __init__(
            self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None
    ) -> None:
        import lmdb

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "_cache_" + "".join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.length


class LSUN(VisionDataset):
    """`LSUN <https://www.yf.io/p/lsun>`_ dataset.

    You will need to install the ``lmdb`` package to use this dataset: run
    ``pip install lmdb``

    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
            self,
            root: str,
            classes: Union[str, List[str]] = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.classes = [
            "bedroom",
            "bridge",
            "church_outdoor",
            "classroom",
            "conference_room",
            "dining_room",
            "kitchen",
            "living_room",
            "restaurant",
            "tower",
        ]
        self.semantic_classes = [" ".join(c.split("_")) for c in self.classes]
        self.classes_with_split = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes_with_split:
            self.dbs.append(LSUNClass(root=os.path.join(root, f"{c}_lmdb"), transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes: Union[str, List[str]]) -> List[str]:

        dset_opts = ["train", "val", "test"]

        try:
            classes = cast(str, classes)
            verify_str_arg(classes, "classes", dset_opts)
            if classes == "test":
                classes = [classes]
            else:
                classes = [c + "_" + classes for c in self.classes]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = "Expected type str or Iterable for argument classes, but got type {}."
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr_type = "Expected type str for elements in argument classes, but got type {}."
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr_type.format(type(c)))
                c_short = c.split("_")
                category, dset_opt = "_".join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class", iterable_to_str(self.classes))
                verify_str_arg(category, valid_values=self.classes, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        return "Classes: {classes}".format(**self.__dict__)


class OodLSUN(LSUN):
    def __init__(self, data_path, transform, train, templates=None):

        self.download = True
        self.root = os.path.join(data_path, 'lsun')
        if download:
            if not os.listdir(self.root):
                self._download()
            else:
                _logger.info("LSUN already downloaded")
        super(OodLSUN, self).__init__(root=os.path.join(data_path, 'lsun'),
                                      classes='train' if train else 'val',
                                      transform=transform)

        self.templates = templates if templates else imagenet_templates
        self.idx_to_class = {i: cls for (i, cls) in enumerate(self.classes)}
        self.class_to_idx = {value: key for (key, value) in self.idx_to_class.items()}
        self.targets = self.set_targets()

    def _download(self, category=None):
        categories = list_categories()
        if category is None:
            _logger.info('Downloading', len(categories), 'categories')
            for category in categories:
                download(self.root, category, 'train')
                download(self.root, category, 'val')
            download(self.root, '', 'test')
        else:
            if category == 'test':
                download(self.root, '', 'test')
            elif category not in categories:
                _logger.error(f'{category}, doesn\'t exist in LSUN release')
            else:
                download(self.root, category, 'train')
                download(self.root, category, 'val')

    def set_targets(self):
        _logger.warning("setting targets only for validation set".upper())
        num_ind = int(self.length / len(self.classes))
        targs = [num_ind * [i] for i in range(len(self.classes))]
        return np.array([item for sublist in targs for item in sublist])


def main():
    name = "LSUN"
    dataset = OodLSUN
    run_full_distances(name, dataset, lsun=True)


if __name__ == '__main__':
    main()

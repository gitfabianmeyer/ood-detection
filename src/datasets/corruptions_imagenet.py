import logging
import random

from datasets.imagenet import OodTinyImageNet
from datasets.corruptions import Corruptions

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class OodImagenetO(OodTinyImageNet):
    def __init__(self, data_path, transform, train, method=None, severity=5, templates=None):
        super(OodImagenetO, self).__init__(data_path, transform, train, templates=templates)
        self.corruptions = Corruptions
        self.idx_to_corruption = {i: corr for (i, corr) in enumerate(self.corruptions.keys())}
        self.severity = severity
        self.method = self.get_method(method)
        self.method = self.corruptions[self.idx_to_corruption[method]]

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            img = self.method(img, self.severity)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_method(self, method):
        if method is None:
            method = random.randint(0, len(self.corruptions) - 1)
        return self.corruptions[self.idx_to_corruption[method]]

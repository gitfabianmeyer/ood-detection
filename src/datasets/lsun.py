import os
import logging

import clip
import torchvision
import subprocess
from urllib.request import Request, urlopen

from datasets.classnames import imagenet_templates
from metrics.distances import get_distances_for_dataset
from ood_detection.classification_utils import full_batch_classification
from ood_detection.config import Config

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


class OodLSUN(torchvision.datasets.LSUN):
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


def main():
    data_path = Config.DATAPATH
    train = False
    clip_model, transform = clip.load(Config.VISION_MODEL)

    dataset = OodLSUN(data_path, transform, train)
    # get_distances_for_dataset(dataset, clip_model, "LSUN")
    full_batch_classification(dataset, clip_model, 'LSUN')


if __name__ == '__main__':
    main()

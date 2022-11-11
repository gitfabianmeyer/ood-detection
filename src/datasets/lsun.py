import os

import clip
import torchvision

from src.datasets.zoc_loader import single_isolated_class_loader
from src.ood_detection.config import Config

import subprocess
from urllib.request import Request, urlopen

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


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
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)


class OodLSUN(torchvision.datasets.LSUN):
    def __init__(self, datapath, transform, train):

        self.download = True
        self.root = os.path.join(datapath, 'lsun')
        if download:
            if not os.listdir(self.root):
                self._download()
            else:
                print("LSUN already downloaded")
        super(OodLSUN, self).__init__(root=os.path.join(datapath, 'lsun'),
                                      classes='train' if train else 'val',
                                      transform=transform)
    def _download(self, category=None):
        categories = list_categories()
        if category is None:
            print('Downloading', len(categories), 'categories')
            for category in categories:
                download(self.root, category, 'train')
                download(self.root, category, 'val')
            download(self.root, '', 'test')
        else:
            if category == 'test':
                download(self.root, '', 'test')
            elif category not in categories:
                print('Error:', category, "doesn't exist in", 'LSUN release')
            else:
                download(self.root, category, 'train')
                download(self.root, category, 'val')


def main():
    datapath = Config.DATAPATH
    train = False
    _, transform = clip.load(Config.VISION_MODEL)
    cifar = OodLSUN(datapath, transform, train)
    loaders = single_isolated_class_loader(cifar)

    for loader in loaders.keys():
        print(loader)
        for item in loaders[loader]:
            print(10)
            pass


if __name__ == '__main__':
    main()

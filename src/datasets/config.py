import collections

from datasets.caltech import OodCaltech101
from datasets.caltech_cub import OodCub2011
from datasets.cifar import OodCifar10
from datasets.cifar100 import OodCifar100
from datasets.dtd import OodDTD
from datasets.fashion_mnist import OodFashionMNIST
from datasets.flowers102 import OodFlowers102
from datasets.gtsrb import OodGTSRB
from datasets.imagenet import OodTinyImageNet
from datasets.lsun import OodLSUN
from datasets.mnist import OodMNIST
from datasets.stanfordcars import OodStanfordCars
from datasets.svhn import OodSVHN

DATASETS_DICT = collections.OrderedDict()

DATASETS_DICT["caltech101"] = OodCaltech101
DATASETS_DICT["caltech cub"] = OodCub2011
DATASETS_DICT["cifar10"] = OodCifar10
DATASETS_DICT["cifar100"] = OodCifar100
DATASETS_DICT["dtd"] = OodDTD
DATASETS_DICT["fashion mnist"] = OodFashionMNIST
DATASETS_DICT["flowers102"] = OodFlowers102
DATASETS_DICT["gtsrb"] = OodGTSRB
DATASETS_DICT["imagenet"] = OodTinyImageNet
DATASETS_DICT["lsun"] = OodLSUN
DATASETS_DICT["mnist"] = OodMNIST
DATASETS_DICT["stanford cars"] = OodStanfordCars
DATASETS_DICT["svhn"] = OodSVHN

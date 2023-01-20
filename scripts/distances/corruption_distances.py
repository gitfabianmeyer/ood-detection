import os

from datasets.config import DATASETS_DICT
from metrics.distances import run_full_distances

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for dname, dset in DATASETS_DICT.items():

    if dname == 'lsun':
        lsun = True
    else:
        lsun = False
    run_full_distances(dname, dset, lsun)

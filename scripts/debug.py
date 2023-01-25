from collections import defaultdict

import numpy as np

import wandb
from datasets.config import DATASETS_DICT
from ood_detection.config import Config
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def main():
    for dname, dset in DATASETS_DICT.items():
        # run = wandb.init(project="thesis-classlengths",
        #                  entity="wandbefab",
        #                  name=dname,
        #                  tags=['zeroshot',
        #                        'zsa'])
        print(f"\n\n----------------------------------- {dname}----------------------------------- ")
        for split in ['train', 'val', 'test']:
            lengths = defaultdict(int)

            print(f"\t {split}")
            dataset = dset(Config.DATAPATH,
                           split=split,
                           transform=None)

            for target in dataset.targets:
                lengths[dataset.idx_to_class[int(target)] + f'-{split}'] += 1


            smallest = np.inf
            sk = "NONE"
            for key, value in lengths.items():
                if value < smallest:
                    smallest = value
                    sk = key

            print(f"smallest: {sk} with {smallest}")
        # wandb.log(lengths)
        # run.finish()


if __name__ == '__main__':
    main()

from collections import defaultdict

import numpy as np
import wandb

import wandb
from datasets.caltech_cub import OodCub2011
from datasets.gtsrb import OodGTSRB
from datasets.svhn import OodSVHN
from ood_detection.config import Config
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms

from datasets.config import DATASETS_DICT


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def main():
    for dname, dset in DATASETS_DICT.items():

        run = wandb.init(project="thesis-dataset_sizes",
                         entity="wandbefab",
                         name=dname,
                         tags=["stats"])
        for split in ["train", "test", "val"]:
            min_size, max_size = 10000, 0
            counts = defaultdict(int)

            dataset = dset(Config.DATAPATH,
                           split=split,
                           transform=transforms.Compose([_convert_image_to_rgb, ToTensor()]))

            loader = DataLoader(dataset,
                                batch_size=1)
            print(f"{split}: {len(loader)}")

            for image, target in loader:
                height, width = image.squeeze().shape[1:]
                if height < min_size:
                    min_size = height
                if height > max_size:
                    max_size = height
                if width < min_size:
                    min_size = width
                if width > max_size:
                    max_size = width

                counts[int(target)] += 1
            results = {f"{split}_classes": len(counts.keys()), f"{split}_images": sum(counts.values()),
                       f"{split}_mean_per_class": sum(counts.values()) / len(counts.keys()),
                       f"{split}_smallest_edge": min_size, f"{split}_largest_edge": max_size}
            wandb.log(results)
        run.finish()


if __name__ == '__main__':
    main()

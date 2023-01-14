from collections import defaultdict

import numpy as np
from datasets.caltech_cub import OodCub2011
from datasets.gtsrb import OodGTSRB
from datasets.svhn import OodSVHN
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.ood_detection.config import Config

dset = OodGTSRB

min_size, max_size = 10000, 0
counts = defaultdict(int)
for lab in [True, False]:
    dataset = dset(Config.DATAPATH,
                   train=True,
                   transform=ToTensor())

    loader = DataLoader(dataset,
                        batch_size=1)
    print(f"{lab}: {len(loader)}")

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

length = len(counts.keys())
summ = sum(counts.values())
means = summ / length
print(f"num classes: {length}")
print(f"Sum images: {summ}")
print(f"means: {means}")

print(f"Min:{min_size}")
print(f"Max: {max_size}")

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import clip
import torch


size(0)
top1 = (top1 / n) * 100
top5 = (top5 / n) * 100

print(f"\nClip Top1 Acc: {top1:.2f} with zeroshot")
print(f"\nClip Top5 Acc: {top5:.2f} with zeroshot")

print("Done")


if __name__ == '__main__':
    main()

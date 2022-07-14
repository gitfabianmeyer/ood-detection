import clip
import torch
from ood_detection import ood_classification
import torchvision

from ood_detection.config import Config

from ood_detection.datasets.flowers102 import FlowersWithLabels
from ood_detection.datasets.stanfordcars import StandardizedStanfordCars
from ood_detection.ood_classification import prep_subset_image_files, get_dataset_features

from src.ood_detection.datasets.caltech import StandardizedCaltech

datapath = Config.DATAPATH

model, preprocess = clip.load(Config.VISION_MODEL)
model.eval()

calt = StandardizedCaltech(Config.DATAPATH,
                           transform=preprocess)

calt = prep_subset_image_files(calt, 3)

# set label to OOD label from the train set
calt._labels = [38 for _ in range(len(calt._labels))]
ood_loader = torch.utils.data.DataLoader(calt,
                                         batch_size=16,
                                         num_workers=8,
                                         shuffle=False
                                         )

ood_features, ood_labels = get_dataset_features(ood_loader,
                                                model,
                                                './',
                                                './')

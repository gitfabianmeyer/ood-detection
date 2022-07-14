import clip
import torch
from ood_detection import ood_classification
import torchvision

from ood_detection.config import Config

from ood_detection.datasets.flowers102 import FlowersWithLabels
from ood_detection.datasets.stanfordcars import StandardizedStanfordCars
from ood_detection.ood_classification import prep_subset_image_files, get_dataset_features

datapath = Config.DATAPATH

model, preprocess = clip.load(Config.VISION_MODEL)
model.eval()

cars = StandardizedStanfordCars(Config.DATAPATH,
                                transform=preprocess)

cars = prep_subset_image_files(cars, 3)

# set label to OOD label from the train set
cars._labels = [38 for _ in range(len(cars._labels))]
ood_loader = torch.utils.data.DataLoader(cars,
                                         batch_size=16,
                                         num_workers=8,
                                         shuffle=False
                                         )

ood_features, ood_labels = get_dataset_features(ood_loader,
                                                model,
                                                './',
                                                './')
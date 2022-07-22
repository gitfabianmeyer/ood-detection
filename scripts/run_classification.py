import clip
from ood_detection import ood_classification
import torchvision

from ood_detection.config import Config

from ood_detection.datasets.flowers102 import FlowersWithLabels
from ood_detection.datasets.stanfordcars import StandardizedStanfordCars

from ood_detection.datasets.caltech import StandardizedCaltech
from itertools import chain, combinations

"""
Script to run ood classification.
"""


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))


datapath = Config.DATAPATH

datasets = {}

model, preprocess = clip.load(Config.VISION_MODEL)
model.eval()

pets = torchvision.datasets.OxfordIIITPet(Config.DATAPATH,
                                          transform=preprocess,
                                          download=True)

cars = StandardizedStanfordCars(Config.DATAPATH,
                                transform=preprocess)

flowers = FlowersWithLabels(Config.DATAPATH,
                            transform=preprocess)

dtd = torchvision.datasets.DTD(Config.DATAPATH,
                               transform=preprocess,
                               download=True)

aircraft = torchvision.datasets.FGVCAircraft(Config.DATAPATH,
                                             transform=preprocess,
                                             download=True)
caltech = StandardizedCaltech(Config.DATAPATH,
                              transform=preprocess,
                              download=True)
print("Loaded sets...")
dataset_list = [{'oxfordpets': pets},
                {'stanfordcars': cars},
                {'flowers': flowers},
                {"dtd": dtd},
                {'fcgvaaircraft': aircraft},
                {'caltech': caltech}]

for i, data in enumerate(dataset_list):
    data_dict = {}
    rest_list = dataset_list[:i] + dataset_list[i + 1:]

    data_dict["id"] = data
    for combo in all_subsets(rest_list):
        data_dict["ood"] = {}
        for element in combo:
            element_name = list(element.keys())[0]
            data_dict["ood"][element_name]

        # do the classification for all possible combis
        ood_classification.main(datasets)

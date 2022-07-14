import clip
from ood_detection import ood_classification
import torchvision

from ood_detection.config import Config

from ood_detection.datasets.flowers102 import FlowersWithLabels
from ood_detection.datasets.stanfordcars import StandardizedStanfordCars

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


print("loaded sets")
datasets["id"] = {'oxfordpets': pets}
datasets["ood"] = {}
datasets["ood"]["stanfordcars"] = cars
datasets["ood"]["flowers"] = flowers
ood_classification.main(datasets)

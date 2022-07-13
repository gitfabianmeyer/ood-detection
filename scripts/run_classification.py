import clip
from ood_detection import ood_classification
import torchvision

from ood_detection.config import Config

datapath = Config.DATAPATH

datasets = {}

model, preprocess = clip.load(Config.VISION_MODEL)
model.eval()

pets = torchvision.datasets.OxfordIIITPet(Config.DATAPATH,
                                          transform=preprocess,
                                          download=True)

cars = torchvision.datasets.StanfordCars(Config.DATAPATH,
                                         transform=preprocess,
                                         download=True)
flowers = torchvision.datasets.Flowers102(Config.DATAPATH,
                                          transform=preprocess,
                                          download=True)

datasets["id"] = {'oxfordpets': pets}
datasets["ood"] = {}
datasets["ood"]["stanfordcars"] = cars
datasets["ood"]["flowers"] = flowers

ood_classification.main(datasets)

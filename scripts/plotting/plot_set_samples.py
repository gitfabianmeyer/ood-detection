import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import clip
import torch
import torchvision
from ood_detection.config import Config
from ood_detection.datasets.caltech import StandardizedCaltech
from ood_detection.datasets.flowers102 import FlowersWithLabels
from ood_detection.datasets.stanfordcars import StandardizedStanfordCars
from ood_detection.ood_classification import prep_subset_images
from torch.utils.data import DataLoader
from tqdm import tqdm

device = Config.DEVICE
model, preprocess = clip.load(Config.VISION_MODEL, device=device)

flowers = FlowersWithLabels(Config.DATAPATH, preprocess)
cars = StandardizedStanfordCars(Config.DATAPATH, transform=preprocess)
dtd = torchvision.datasets.DTD(Config.DATAPATH, transform=preprocess)
caltech = StandardizedCaltech(Config.DATAPATH, preprocess)
aircraft = torchvision.datasets.FGVCAircraft(Config.DATAPATH, transform=preprocess, download=True)
pets = torchvision.datasets.OxfordIIITPet(Config.DATAPATH, transform=preprocess)
sets = {
    'flowers': flowers,
    'cars': cars,
    'dtd': dtd,
    'caltech': caltech,
    'aircraft': aircraft,
    'pets': pets
}


def main():
    for name, dataset in sets.items():
        subset = prep_subset_images(dataset, n=1000)
        loader = DataLoader(subset, batch_size=256)
        features = []
        labels = []
        with torch.no_grad():
            for images, samples in tqdm(loader):
                images = images.to(device)
                target = target.to(device)
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)
            features = torch.cat(features)
            labels = torch.cat(labels)

        path = os.path.join(Config.DATAPATH, 'samples')
        os.makedirs(path, exist_ok=True)

        features_path = os.path.join(path, name + '_features.pt')
        labels_path = os.path.join(path, name + 'labels.pt')
        torch.save(features, features_path)
        torch.save(labels, labels_path)
        print(f"Saved {name} with len {len(features)}")
    print('Done')


if __name__ == '__main__':
    main()

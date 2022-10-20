import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import clip
import torch
from tqdm import tqdm

from ood_detection.config import Config
from ood_detection.classnames import imagenet_templates
from ood_detection.ood_utils import zeroshot_classifier

from zoc.dataloaders.aircrat_loader import aircraft_single_isolated_class_loader
from zoc.dataloaders.cars_loader import cars_single_isolated_class_loader
from zoc.dataloaders.cifar10 import cifar10_single_isolated_class_loader
from zoc.dataloaders.dtd_loader import dtd_single_isolated_class_loader
from zoc.dataloaders.gtsrb_loader import gtsrb_single_isolated_class_loader
from zoc.dataloaders.oxford3pets_loader import oxford3_single_isolated_class_loader

device = Config.DEVICE
batch_size = 128
datasets = {'gtsrb': gtsrb_single_isolated_class_loader(batch_size),
            'aircraft': aircraft_single_isolated_class_loader(batch_size),
            'cars': cars_single_isolated_class_loader(batch_size),
            'cifar10': cifar10_single_isolated_class_loader(batch_size),
            'dtd': dtd_single_isolated_class_loader(batch_size),
            'pets': oxford3_single_isolated_class_loader(batch_size)
            }
dataset_labels = []

clip_model, preprocess = clip.load('ViT-B/32')
for name, isolate_loader in datasets.items():

    dataset_label = list(isolate_loader.keys())
    base_path = os.path.join(Config.DATAPATH, 'features')
    data_path = os.path.join(base_path, name)
    os.makedirs(data_path, exist_ok=True)
    for label in dataset_label:
        loader = isolate_loader[label]
        if type(label) is str and "/" in label:
            label = label.replace("/", '+')
        label_path = os.path.join(data_path, str(label) + '.pt')
        with torch.no_grad():
            features = []
            for images in tqdm(loader):
                images = images.to(device)
                batch_features = clip_model.encode_image(images)
                batch_features /= batch_features.norm(dim=1, keepdim=True)
                features.append(batch_features)

            features = torch.cat(features)

        torch.save(features, label_path)
        print(f"Stored: {name} / {label} ({features.shape} images x features) to {label_path}")

    # encode and store labels
    label_features = zeroshot_classifier(dataset_label, imagenet_templates, clip)

    labels_path = os.path.join(data_path, 'labels.pt')
    torch.save(label_features, labels_path)
    print(f"Stored label features for {name} ({len(dataset_label)} labels ) in {labels_path}")

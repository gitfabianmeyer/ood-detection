import argparse
import os.path
import random
from collections import defaultdict
from datetime import datetime

import clip
import torch
import torchvision
from ood_detection.ood_utils import zeroshot_classifier, classify
from tqdm import tqdm

from ood_detection.classnames import fgvcaircraft_classes, \
    oxfordpets_classes, \
    imagenet_templates, \
    stanfordcars_classes, \
    flowers_classes, \
    caltech101_classes, \
    dtd_classes

from ood_detection.config import Config

from ood_detection.plotting.distributions import plot_pca_analysis

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def prep_subset_images(dataset: torchvision.datasets, n):
    split_by_label_dict = defaultdict(list)

    # just use some imgs for each label
    for i in range(len(dataset._images)):
        split_by_label_dict[dataset._labels[i]].append(dataset._images[i])

    imgs = []
    targets = []

    for label, items in split_by_label_dict.items():
        imgs = imgs + random.sample(items, n)
        targets = targets + [label for _ in range(n)]

    dataset._images = imgs
    dataset._labels = targets

    return dataset


def prep_subset_image_files(dataset: torchvision.datasets, n):
    split_by_label_dict = defaultdict(list)

    # just use some imgs for each label
    for i in range(len(dataset._image_files)):
        split_by_label_dict[dataset._labels[i]].append(dataset._image_files[i])
    imgs = []
    targets = []

    for label, items in split_by_label_dict.items():
        if n > len(items):
            n = len(items)
        imgs = imgs + random.sample(items, n)
        targets = targets + [label for _ in range(n)]

    dataset._image_files = imgs
    dataset._labels = targets

    return dataset


def get_dataset_features(loader: torch.utils.data.DataLoader, model, features_path, targets_path):
    features = []
    labels = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.to(device)
            target = target.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
        features = torch.cat(features)

        labels = torch.cat(labels)

        torch.save(features, features_path)
        torch.save(labels, targets_path)
        return features, labels


def get_classnames_from_vision_set(vision_set: torchvision.datasets):
    return vision_set.classes


def main(dataset_dictionary):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=float, default=50, help='ns')
    parser.add_argument('--load_data', type=bool, default=False, help='ld')
    parser.add_argument('--use_aircraft', type=bool, default=True, help='ua')
    parser.add_argument('--use_cars', type=bool, default=True, help='uc')
    parser.add_argument('--use_flowers', type=bool, default=True, help='uf')
    parser.add_argument('--plot', type=bool, default=True, help='pl')
    parser.add_argument('--batch_size', type=int, default=64, help='bs')
    args = parser.parse_args()

    run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    curr_datapath = os.path.join(Config.FEATURES, run)

    model, preprocess = clip.load(Config.VISION_MODEL)
    model.eval()

    ood_dict = dataset_dictionary['ood']
    id_dict = dataset_dictionary["id"]
    id_images = list(id_dict.values())[0]

    id_images = prep_subset_images(id_images, args.num_samples)
    id_loader = torch.utils.data.DataLoader(id_images,
                                            batch_size=args.batch_size,
                                            num_workers=8,
                                            shuffle=False)
    # add OOD label to the data
    id_images.class_to_idx["OOD"] = len(id_images.classes)
    id_images.classes.append("OOD")

    os.makedirs(curr_datapath, exist_ok=True)

    id_name = list(id_dict.keys())[0]
    print(f"Starting run with In-Distribution set {id_name}")
    id_features_path = os.path.join(curr_datapath, f"{id_name}_{len(id_images)}_f.pt")
    id_targets_path = os.path.join(curr_datapath, f"{id_name}_{len(id_images)}_t.pt")

    # get the features for each label ( including the OOD label )
    id_classnames = get_classnames_from_vision_set(list(id_dict.values())[0])
    ood_classnames = ood_dict.keys()
    zeroshot_weights = zeroshot_classifier(id_classnames, ood_classnames, imagenet_templates, model)

    # obtain & save features
    if not args.load_data:
        test_features, test_labels = get_dataset_features(id_loader, model, id_features_path,
                                                          id_targets_path)
    else:
        test_features = torch.load(id_features_path)
        test_labels = torch.load(id_targets_path)

    print("In distribution classification:")
    classify(test_features, zeroshot_weights, test_labels, id_name)

    # for OOD in the 1-vs-all case (n wrong classes, 1 OOD class)
    print("\nOut of distribution classification...")

    for name, dataset in ood_dict.items():
        ood_features_path = os.path.join(curr_datapath, f"{name}_{len(dataset)}_f.pt")
        ood_targets_path = os.path.join(curr_datapath, f"{name}_{len(dataset)}_t.pt")

        ood_images = prep_subset_image_files(dataset, args.num_samples)

        # set label to OOD label from the train set
        ood_images._labels = [id_images.class_to_idx["OOD"] for _ in range(len(ood_images._labels))]
        ood_loader = torch.utils.data.DataLoader(ood_images,
                                                 batch_size=args.batch_size,
                                                 num_workers=8,
                                                 shuffle=False
                                                 )


        ood_features, ood_labels = get_dataset_features(ood_loader,
                                                        model,
                                                        ood_features_path,
                                                        ood_targets_path)

        classify(ood_features, zeroshot_weights, ood_labels, name)
        print(80 * "-")

    if args.plot:
        plot_pca_analysis(curr_datapath)

    return True

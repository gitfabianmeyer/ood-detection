import argparse
import os.path
import random
from collections import defaultdict
import datetime
from typing import Dict

import clip
import torch
import torchvision
from tqdm import tqdm

from ood_detection.classnames import fgvcaircraft_classes, \
    oxford_pets_classes, \
    imagenet_templates, \
    stanford_classes, \
    flowers_classes, \
    caltech101_classes

from ood_detection.config import Config

from ood_detection.plotting.distributions import plot_pca_analysis

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_ood_targets(ood_names, clip_model, templates):
    # TODO here add more classes for OOD
    # collect label embeddings from another dataset plan and simple, maybe just a random vector in clip space
    classnames = []
    for ood_name in ood_names:
        if ood_name == "fgvcaircraft":
            print(f"Using {ood_name} as ood")
            classnames.extend(fgvcaircraft_classes)

        if ood_name == "stanford":
            print(f"Using {ood_name} as ood")
            classnames.extend(stanford_classes)

        if ood_name == 'caltech101':
            print(f"Using {ood_name} as ood")
            classnames.extend(caltech101_classes)
    print(f'Number of classes merged to OOD label: {len(classnames)} from {len(ood_names)} sets')

    embedding = []
    for classname in classnames:
        class_embeddings = get_normed_embeddings(classname, clip_model, templates)
        embedding.append(class_embeddings)
    # take the center of this OOD
    embedding = torch.stack(embedding)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.mean(dim=0)
    embedding /= embedding.norm()
    return embedding


def zeroshot_classifier(classnames: list, templates: list, clip_model):
    with torch.no_grad():
        weights = []
        for classname in classnames:
            if classname == 'OOD':
                print("Found OOD")
                continue
            class_embeddings = get_normed_embeddings(classname, clip_model, templates)
            weights.append(class_embeddings)

        # last class is the ood class
        weights.append(get_ood_targets(clip_model, templates))

        weights = torch.stack(weights, dim=1).to(device)
        return weights


def get_normed_embeddings(classname, clip_model, templates):
    texts = [template.format(classname) for template in templates]
    texts = clip.tokenize(texts).to(device)
    # casual normalization stuff, stolen from tip adapter paper
    class_embeddings = clip_model.encode_text(texts)  # embed
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embeddings = class_embeddings.mean(dim=0)
    class_embeddings /= class_embeddings.norm()
    return class_embeddings


def accuracy(output, target, top_k=(1,)):
    pred = output.topk(max(top_k), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # TODO here you get the the class predictions

    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in top_k]


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


def classify(features, zeroshot_weights, labels, dataset):
    top1, top5, n = 0., 0., 0.
    logits = 100. * features @ zeroshot_weights
    acc1, acc5 = accuracy(logits, labels, top_k=(1, 5))
    top1 += acc1
    top5 += acc5
    n = features.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    print(f"\nClip Top1 Acc: {top1:.2f} with zeroshot on {dataset} ({features.size(0)} images)")
    print(f"\nClip Top5 Acc: {top5:.2f} with zeroshot on {dataset}")


def get_classnames_from_vision_set(vision_set: torchvision.datasets):
    return vision_set.classes



def main(dataset_dictionary):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=float, default=50, help='ns')
    parser.add_argument('--vision_model', type=str, default='RN50', help='vm')
    parser.add_argument('--load_data', type=bool, default=False, help='ld')
    parser.add_argument('--use_aircraft', type=bool, default=True, help='ua')
    parser.add_argument('--use_cars', type=bool, default=True, help='uc')
    parser.add_argument('--use_flowers', type=bool, default=True, help='uf')
    parser.add_argument('--plot', type=bool, default=True, help='pl')
    args = parser.parse_args()


    run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    curr_datapath = os.path.join(Config.FEATURES, run)

    model, preprocess = clip.load(args.vision_model)
    model.eval()

    id_dict = dataset_dictionary["id"]
    id_images = id_dict.values()[0]

    #id_images = torchvision.datasets.OxfordIIITPet(curr_datapath,
    #                                               split='trainval',
    #                                               transform=preprocess,
    #                                               download=True)

    id_images = prep_subset_images(id_images, args.num_samples)
    id_loader = torch.utils.data.DataLoader(id_images,
                                                       batch_size=16,
                                                       num_workers=8,
                                                       shuffle=False)
    # add OOD label to the data
    id_images.class_to_idx["OOD"] = len(id_images.classes)
    id_images.classes.append("OOD")

    os.makedirs(curr_datapath, exist_ok=True)

    id_name =  id_dict.keys()[0]
    print(f"Starting run with In-Distribution set {id_name}")
    id_features_path = os.path.join(curr_datapath, f"{id_name}_{len(id_images)}_f.pt")
    id_targets_path = os.path.join(curr_datapath, f"{id_name}_{len(id_images)}_t.pt")


    # get the features for each label ( including the OOD label )
    id_classnames = get_classnames_from_vision_set(id_dict.values()[0])
    zeroshot_weights = zeroshot_classifier(id_classnames, imagenet_templates, model)

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

    for name, dataset in dataset_dictionary["ood"]:
        ood_features_path = os.path.join(curr_datapath, f"{name}_{len(dataset)}_f.pt")
        ood_targets_path = os.path.join(curr_datapath, f"{name}_{len(dataset)}_t.pt")

        ood_images = prep_subset_image_files(dataset, args.num_samples)

        # set label to OOD label from the train set
        ood_images._labels = [id_images.class_to_idx["OOD"] for _ in range(len(ood_images._labels))]
        ood_loader = torch.utils.data.DataLoader(ood_images)

        ood_features, ood_labels = get_dataset_features(ood_loader,
                                                        model,
                                                        ood_features_path,
                                                        ood_targets_path)
        classify(ood_features, zeroshot_weights, ood_labels, name)
        print(80* "-")

    if args.plot:
        plot_pca_analysis(curr_datapath)

    print("Finished zeroshot classification")


if __name__ == '__main__':
    main()

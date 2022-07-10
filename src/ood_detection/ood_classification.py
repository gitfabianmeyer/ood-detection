import os.path
import random
from collections import defaultdict

import clip
import torch
import torchvision
from tqdm import tqdm

from src.ood_detection.classnames import fgvcaircraft_classes, oxford_pets_classes, imagenet_templates

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
datapath = '/mnt/c/Users/fmeyer/Git/ood-detection/data'
pets_features_path = os.path.join(datapath, "pets_f_test.pt")
pets_targets_path = os.path.join(datapath, "pets_t_test.pt")
num_samples = 2

print(clip.available_models())
VISION = 'RN50'
# ALWAYS SET TO FALSE WHEN USING NEW VISION MODEL
load_test = False


def get_ood_targets(clip_model, templates):
    # collect label embeddings from another dataset plan and simple, maybe just a random vector in clip space
    classnames = fgvcaircraft_classes
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


def prep_subset_images(dataset: torchvision.datasets, n=num_samples):
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


def prep_subset_image_files(dataset: torchvision.datasets, n=num_samples):
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


def get_dataset_features(loader: torch.utils.data.DataLoader, features_path, targets_path):
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


model, preprocess = clip.load(VISION)
model.eval()

oxfordiiipets_images = torchvision.datasets.OxfordIIITPet(datapath,
                                                          split='trainval',
                                                          transform=preprocess,
                                                          download=True)

oxfordiiipets_images = prep_subset_images(oxfordiiipets_images)
oxfordiiipets_loader = torch.utils.data.DataLoader(oxfordiiipets_images,
                                                   batch_size=num_samples,
                                                   num_workers=8,
                                                   shuffle=False)
random.seed(42)
torch.manual_seed(42)
# get the features for each label ( including the OOD label )
zeroshot_weights = zeroshot_classifier(oxford_pets_classes, imagenet_templates, model)
# add OOD label to the data, even if not really needed
oxfordiiipets_images.class_to_idx["OOD"] = len(oxfordiiipets_images.classes)
oxfordiiipets_images.classes.append("OOD")

# obtain & save features
if not load_test:
    test_features, test_labels = get_dataset_features(oxfordiiipets_loader, pets_features_path, pets_targets_path)
else:
    test_features = torch.load(pets_features_path)
    test_labels = torch.load(pets_targets_path)


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


print("In distribution classification:")
classify(test_features, zeroshot_weights, test_labels, "oxford pets")

# for OOD in the 1-vs-all case (n wrong classes, 1 OOD class)
print("\nOut of distribution classification")
dtd_features_path = os.path.join(datapath, "dtd_f_test.pt")
dtd_targets_path = os.path.join(datapath, "dtd_t_test.pt")

dtd_images = torchvision.datasets.DTD(datapath,
                                      split='train',
                                      transform=preprocess,
                                      download=True)
dtd_images = prep_subset_image_files(dtd_images)
# set label to OOD label from the train set
# TODO check if worked
dtd_images._labels = [oxfordiiipets_images.class_to_idx["OOD"] for i in range(len(dtd_images._labels))]
dtd_loader = torch.utils.data.DataLoader(dtd_images)

# get img_features and targets
load_dtd = False
if not load_dtd:
    dtd_features, dtd_labels = get_dataset_features(dtd_loader, dtd_features_path, dtd_targets_path)
else:
    dtd_features = torch.load(dtd_features_path)
    dtd_labels = torch.load(dtd_targets_path)

classify(dtd_features, zeroshot_weights, dtd_labels, "dtd")
print("DONE")

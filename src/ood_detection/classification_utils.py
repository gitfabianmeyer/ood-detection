import logging

import clip
import numpy as np
import torch
from metrics.distances_utils import shape_printer
from ood_detection.config import Config
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from zeroshot.classification import get_cosine_similarity_matrix_for_normed_features

device = Config.DEVICE
_logger = logging.getLogger(__name__)


def full_classification(dataset, model, name):
    dataloader = DataLoader(dataset, batch_size=512)
    features, targets = get_dataset_features(dataloader, model, None, None)
    templates = dataset.templates
    print(templates)
    zeroshot_weights = zeroshot_classifier(dataset.classes,
                                           templates=templates,
                                           clip_model=model)
    classify(features, zeroshot_weights, targets, name, True)


@torch.no_grad()
def full_batch_classification(dataset, model, name):
    dataloader = DataLoader(dataset, batch_size=10)
    templates = dataset.templates
    zeroshot_weights = zeroshot_classifier(dataset.classes,
                                           templates=templates,
                                           clip_model=model)

    accuracies1 = []

    for images, targets in tqdm(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        top1 = classify(image_features, zeroshot_weights, targets)
        accuracies1.append(top1)

    mean1 = np.mean(accuracies1)

    print(f"\nClip Top1 Acc: {mean1:.3f} with zeroshot on {name} ")


@torch.no_grad()
def get_dataset_features(loader: torch.utils.data.DataLoader, model, features_path=None, targets_path=None):
    features, labels = [], []
    _logger.info(f"Getting dataset features...")
    for i, (images, target) in enumerate(tqdm(loader)):
        images = images.to(device)
        target = target.to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        features.append(image_features)
        labels.append(target)
    features = torch.cat(features)

    labels = torch.cat(labels)

    if features_path and targets_path:
        torch.save(features, features_path)
        torch.save(labels, targets_path)
    return features.to(torch.float32), labels.to(torch.float32)


@torch.no_grad()
def zeroshot_classifier(classnames: list, templates: list, clip_model):
    weights = []
    for classname in classnames:
        class_embeddings = get_normed_classname_embedding(classname, clip_model, templates)
        weights.append(class_embeddings)

    return torch.stack(weights).to(torch.float32)


def classify(features, zeroshot_weights, targets, dataset=None, print_results=False):
    top1, top5, n = 0., 0., 0.,
    logits = get_cosine_similarity_matrix_for_normed_features(features, zeroshot_weights, 1)
    acc1 = accuracy(logits, targets, top_k=(1,))[0]
    top1 += acc1
    n = features.size(0)
    top1 = (top1 / n)
    if print_results and dataset:
        print(f"\nClip Top1 Acc: {top1:.3f} with zeroshot on {dataset} ({features.size(0)} images)")
    return top1


def accuracy(output, target, top_k=(1,)):
    output = output.cpu()
    target = target.cpu()
    pred = output.topk(max(top_k), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in top_k]


def macro_f1_score(output, targets):
    output = output.cpu()
    targets = targets.cpu()
    pred = output.topk(1, 1, True, True)[1].t()
    f1_score(targets, pred, average='macro')


@torch.no_grad()
def get_normed_classname_embedding(classname, clip_model, templates):
    texts = [template.format(classname) for template in templates]
    texts = clip.tokenize(texts).to(Config.DEVICE)
    # casual normalization stuff, stolen from tip adapter paper
    class_embeddings = clip_model.encode_text(texts).to(torch.float32)  # embed
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embeddings = class_embeddings.mean(dim=0)
    class_embeddings /= class_embeddings.norm()
    return class_embeddings

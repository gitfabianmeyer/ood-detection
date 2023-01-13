import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import clip
import numpy as np
import torch
from datasets import config
from datasets.config import DATASETS_DICT

import wandb
from datasets.cifar import OodCifar10
from ood_detection.config import Config
from ood_detection.classification_utils import accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm

from ood_detection.classification_utils import zeroshot_classifier


def get_std_mean(list_of_tensors):
    try:
        tensors = np.array(torch.stack(list_of_tensors))
        mean = np.mean(tensors)
        std = np.std(tensors)
    except Exception as e:
        mean, std = 0, 0
    return std, mean


@torch.no_grad()
def clip_zeroshot(features, targets, zeroshot_weights, temperature):
    results = {}
    top1_accs = []
    logit_means = []
    all_confidences, confidences_true, confidences_wrong = [], [], []
    logits = temperature * features @ zeroshot_weights.T
    logit_means.append(torch.mean(logits))
    softmaxs = torch.softmax(logits, dim=-1)
    top1_accs.append(accuracy(logits, targets)[0] / len(targets))

    confidences, correct = conf_scores(softmaxs, targets)
    all_confidences.extend(confidences)
    confidences_true.extend(confidences[correct.T])
    confidences_wrong.extend(confidences[~correct.T])

    all_confidences = get_std_mean(all_confidences)
    confidences_true = get_std_mean(confidences_true)
    confidences_wrong = get_std_mean(confidences_wrong)
    logits = get_std_mean(logit_means)
    accuracies = np.mean(top1_accs) if len(top1_accs) > 0 else 0

    results["confidences_std"], results["confidences"] = all_confidences
    results["correct_std"], results["correct"] = confidences_true
    results["false_std"], results["false"] = confidences_wrong
    results['temperature'] = temperature
    results["logits_std"], results["logits"] = logits
    results["accuracy"] = accuracies
    return results


@torch.no_grad()
def get_dataset_features(clip_model, dataloader):
    features, targets = [], []
    for num_batches, (images, targs) in enumerate(tqdm(dataloader)):
        images = images.to(Config.DEVICE)
        targs = targs.to(Config.DEVICE)
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        features.append(image_features)
        targets.append(targs)
    features = torch.cat(features)
    targets = torch.cat(targets)
    return features, targets


def conf_scores(softmax_scores=None, targets=None):
    softmax_scores = softmax_scores.cpu()
    targets = targets.cpu()
    confidences, indice = softmax_scores.topk(1, 1, True, True)
    indice = indice.t()
    correct = indice.eq(targets.view(1, -1).expand_as(indice))
    return confidences, correct


def main():
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    clip_model.eval()

    for dname, dset in DATASETS_DICT.items():
        if dname != 'cifar10':
            continue
        dataset = dset(Config.DATAPATH,
                       train=False,
                       transform=clip_transform)

        dataloader = DataLoader(dataset,
                                batch_size=512)
        print("Generating Features")

        features, targets = get_dataset_features(clip_model, dataloader)
        print("Generating Zeroshot weights")
        zsw = zeroshot_classifier(dataset.classes,
                                  dataset.templates,
                                  clip_model)

        run = wandb.init(project="thesis-temperatures",
                         entity="wandbefab",
                         name=dname,
                         tags=['zeroshot',
                               'zsa',
                               'confidence'])

        for temperature in np.logspace(-7.158429362604483, 6.643856189774724, num=10,
                                       base=2.0):  # 10 values between .007 and 100
            print(f"Running {dname} with temperatur {temperature}")
            results = clip_zeroshot(features, targets, zeroshot_weights=zsw, temperature=temperature)
            wandb.log(results)
        run.finish


if __name__ == '__main__':
    main()

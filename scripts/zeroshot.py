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

    logits = temperature * features.to(torch.float32) @ zeroshot_weights.T.to(torch.float32)

    top1_acc = accuracy(logits, targets)[0] / len(targets)

    softmaxs = torch.softmax(logits, dim=-1)
    confidences, correct = conf_scores(softmaxs, targets)

    results["confidences_std"], results["confidences"] = torch.std_mean(confidences)
    results["correct_std"], results["correct"] = torch.std_mean(confidences[correct.T])
    results["false_std"], results["false"] = torch.std_mean(confidences[~correct.T])
    results['temperature'] = temperature
    results["logits_std"], results["logits"] = torch.std_mean(logits)
    results["accuracy"] = top1_acc
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
    features = torch.cat(features).half()
    targets = torch.cat(targets).half()
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
        print(f"\n\n----------------------------------- {dname}----------------------------------- ")
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

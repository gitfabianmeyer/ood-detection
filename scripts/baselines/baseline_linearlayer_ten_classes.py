import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import numpy as np
from zoc.utils import get_ablation_splits, get_split_specific_targets, fill_auc_lists, fill_f_acc_lists, \
    get_result_mean_dict



from zoc.baseline import linear_layer_detector, get_feature_weight_dict, FeatureSet, train_id_classifier
import argparse
import logging

import clip

import torch
import wandb
from clearml import Task
from datasets.config import DATASETS_DICT
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config

_logger = logging.getLogger(__name__)
splits = [(.4, .6), ]
clearml_model = False
MODEL_PATH = "/home/fmeyer/ZOC/trained_models/COCO/ViT-B32/"


# MODEL_PATH = "/mnt/c/users/fmeyer/git/ood-detection/data/zoc/trained_models/COCO/"


def run_single_dataset_ood(dataset, clip_model, clip_transform, id_classes=.4, runs=5):
    dset = dataset(Config.DATAPATH,
                   split='test',
                   transform=None)
    labels = dset.classes
    id_classes = int(len(labels) * id_classes)
    ood_classes = len(labels) - id_classes
    metrics = linear_layer_detector(dataset, clip_model, clip_transform, id_classes, ood_classes, 5)
    run = wandb.init(project="thesis-zoc_baseline_linear_ten_classes_val_sets",
                     entity="wandbefab",
                     name=dset.name,
                     config={"runs": runs,
                             "id_split": splits[0][0]})
    wandb.log(metrics)
    run.finish()
    return True


def linear_layer_detector(dataset, clip_model, clip_transform, id_classes, ood_classes, runs):
    device = Config.DEVICE
    train_dataset = dataset(Config.DATAPATH,
                            split='train',
                            transform=clip_transform)

    shorted_classes = random.sample(train_dataset.classes, 10)
    train_dataset.classes = shorted_classes

    isolated_classes = IsolatedClasses(train_dataset,
                                       batch_size=512)
    feature_weight_dict_train = get_feature_weight_dict(isolated_classes, clip_model, device)

    val_set = dataset(Config.DATAPATH,
                      split='val',
                      transform=clip_transform),
    val_set.classes = shorted_classes
    isolated_classes = IsolatedClasses(val_set,
                                       batch_size=512
                                       )

    feature_weight_dict_val = get_feature_weight_dict(isolated_classes, clip_model, device)
    ablation_splits = get_ablation_splits(isolated_classes.labels, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum, auc_list_mean, auc_list_max = [], [], []
    for ablation_split in ablation_splits:

        class_to_idx_mapping = {label: i for (i, label) in enumerate(ablation_split)}
        seen_labels = ablation_split[:id_classes]
        unseen_labels = ablation_split[id_classes:]

        _logger.info(f"Seen labels: {seen_labels}\nOOD Labels: {unseen_labels}")

        # train classifier to classify id set
        train_set = FeatureSet(feature_weight_dict_train, seen_labels, class_to_idx_mapping)
        val_set = FeatureSet(feature_weight_dict_val, seen_labels, class_to_idx_mapping)

        linear_layer_run = wandb.init(project="thesis-baseline_linear_clip_training_logs",
                                      entity="wandbefab",
                                      name=train_dataset.name,
                                      tags=[
                                          'linear probe',
                                          'oodd',
                                          'baseline'
                                      ])
        classifier = train_id_classifier(train_set, val_set)
        linear_layer_run.finish()
        _logger.info("Finished finetuning linear layer")
        # eval for ood detection

        isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                                   split='test',
                                                   transform=clip_transform),
                                           batch_size=512)
        feature_weight_dict_test = get_feature_weight_dict(isolated_classes, clip_model, device)

        ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
        f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []
        for i, semantic_label in enumerate(ablation_split):
            # get features
            image_features_for_label = feature_weight_dict_test[semantic_label]
            # calc the logits and softmaxs
            logits = classifier(image_features_for_label.to(torch.float32).to(device))

            assert logits.shape[1] == id_classes
            # detection score is accumulative sum of probs of generated entities
            # careful, only for this setting axis=1
            ood_prob_sum = np.sum(logits.detach().cpu().numpy(), axis=1)
            ood_probs_sum.extend(ood_prob_sum)

            ood_prob_mean = np.mean(logits.detach().cpu().numpy(), axis=1)
            ood_probs_mean.extend(ood_prob_mean)

            top_prob, _ = logits.cpu().topk(1, dim=-1)
            ood_probs_max.extend(top_prob.detach().numpy())

            id_probs_sum.extend(1. - ood_prob_sum)

        targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)
        fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                       targets)
        fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

    metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)

    return metrics


def run_all(args):
    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    for dname, dset in DATASETS_DICT.items():

        # if dname not in ['cifar10', 'caltech cub']:
        #     print(f"Jumping over {dname}")
        #     continue

        _logger.info(f"---------------Running {dname}--------------")

        if dname == 'lsun':
            lsun = True

        else:
            lsun = False

        for split in splits:
            # perform zsoodd
            run_single_dataset_ood(dset,
                                   clip_model=clip_model,
                                   clip_transform=clip_transform,
                                   id_classes=split[0],
                                   runs=args.runs_ood)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_ood', type=int, default=1)

    args = parser.parse_args()
    run_all(args)

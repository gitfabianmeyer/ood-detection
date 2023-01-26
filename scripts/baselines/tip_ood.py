import os

import clip

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch

from adapters.tip_adapter import get_tip_adapter_train_set, get_kshot_train_set, get_cache_model, get_cache_logits, \
    get_train_transform
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config
from zoc.baseline import get_feature_weight_dict, get_zeroshot_weight_dict, sorted_zeroshot_weights
from zoc.utils import get_ablation_splits, get_split_specific_targets, get_auroc_for_max_probs, get_mean_std

import logging

import wandb
from datasets.config import DATASETS_DICT

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

run_clearml = False
runs = 10
kshots = 16
train_epochs = 1
augment_epochs = 10


def main():
    failed = []
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    device = Config.DEVICE

    for dname, dset in DATASETS_DICT.items():
        if dname not in ["fashion mnist", "mnist", "svhn"]:
            continue
        _logger.info(f"\t\tStarting {dname} run...")
        run = wandb.init(project=f"thesis-tip-ood-test",
                         entity="wandbefab",
                         name=dname)
        try:
            results = tip_ood_detector(dset,
                                       clip_model,
                                       clip_transform,
                                       device,
                                       Config.ID_SPLIT,
                                       runs,
                                       kshots,
                                       augment_epochs)
            print(results)
        except Exception as e:
            failed.append(dname)
            raise e

        wandb.log(results)
        run.finish()

    print(f"Failed: {failed}")


def create_tip_train_set(dset, seen_labels, kshots):
    dataset = dset(Config.DATAPATH,
                   transform=get_train_transform(),
                   split='train')

    _logger.info("Creating train set for the seen labels")
    new_class_to_idx = {seen_labels[i]: i for i in range(len(seen_labels))}
    new_idx_to_class = {value: key for (key, value) in new_class_to_idx.items()}

    new_images, new_targets = [], []
    for image, target in zip(dataset.data, dataset.targets):
        old_label = dataset.idx_to_class[int(target)]
        if old_label in seen_labels:
            # get only seen images & new labels for them
            new_images.append(image)
            new_targets.append(new_class_to_idx[old_label])

    dataset.data = new_images
    dataset.targets = np.array(new_targets)
    dataset.idx_to_class = new_idx_to_class
    dataset.class_to_idx = new_class_to_idx
    dataset.classes = seen_labels

    dataset = get_kshot_train_set(dataset, kshots)
    return dataset


def load_hyperparams_from_training(name):
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs('thesis-tip-adapters-16_shots-test')
    for run in runs:
        if run.name == name:
            set_run = run
            break
    df = pd.DataFrame(set_run.history())

    tipf_best_beta = 'tipf_best_beta'
    tipf_best_alpha = 'tipf_best_alpha'
    tip_best_beta = 'tip_best_beta'
    tip_best_alpha = 'tip_best_alpha'
    return {tip_best_alpha: df.iloc[0][tip_best_alpha],
            tip_best_beta: df.iloc[0][tip_best_beta],
            tipf_best_alpha: df.iloc[0][tipf_best_alpha],
            tipf_best_beta: df.iloc[0][tipf_best_beta]}


def tip_ood_detector(dset,
                     clip_model,
                     clip_transform,
                     device,
                     id_classes_split,
                     runs,
                     kshots,
                     augment_epochs):
    dataset = dset(data_path=Config.DATAPATH,
                   split='test',
                   transform=clip_transform)

    # prepare features ...
    isolated_classes = IsolatedClasses(dataset,
                                       batch_size=512,
                                       lsun=False)
    _logger.info('Creating the test weight dicts')
    feature_weight_dict = get_feature_weight_dict(isolated_classes, clip_model, device)
    classes_weight_dict = get_zeroshot_weight_dict(isolated_classes, clip_model)
    _logger.info("Done creating weight dicts.")


    # prepare ablation splits...
    num_id_classes = int(len(dataset.classes) * id_classes_split)
    num_ood_classes = len(dataset.classes) - num_id_classes
    _logger.info(f"ID classes: {num_id_classes}, OOD classes: {num_ood_classes}")
    ablation_splits = get_ablation_splits(dataset.classes, runs, num_id_classes, num_ood_classes)

    # run for the ablation splits
    clip_aucs, tip_aucs = [], []
    for split_idx, split in enumerate(ablation_splits):
        _logger.info(f"Split ({split_idx+1} / {len(ablation_splits)} ")

        seen_labels = split[:num_id_classes]
        unseen_labels = split[num_id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {unseen_labels}")
        zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)
        zeroshot_weights = zeroshot_weights.to(torch.float32)

        # prepare split specific adapter

        # get the kshot train set
        tip_train_set = create_tip_train_set(dset, seen_labels, kshots)
        _logger.info(f"len trainset: {len(tip_train_set)}. Should be: {len(tip_train_set.classes) * kshots} (max)")
        cache_keys, cache_values = get_cache_model(tip_train_set, clip_model, augment_epochs=augment_epochs)
        cache_keys, cache_values = cache_keys.to(torch.float32), cache_values.to(torch.float32)

        hyperparams = load_hyperparams_from_training(dataset.name)
        clip_probs_max, tip_probs_max = [], []

        for split_idx, semantic_label in enumerate(split):
            # get features
            image_features_for_label = feature_weight_dict[semantic_label]
            image_features_for_label = image_features_for_label.to(torch.float32)
            _logger.info(f'image features for label: {image_features_for_label.shape}')
            # calc the logits and softmax
            clip_logits = image_features_for_label @ zeroshot_weights.T
            clip_probs = torch.softmax(clip_logits, dim=-1).squeeze()

            # TIP ADAPTER
            tip_alpha, tip_beta = hyperparams['tip_best_alpha'], hyperparams['tip_best_beta']
            affinity = image_features_for_label @ cache_keys
            cache_logits = get_cache_logits(affinity, cache_values, tip_beta)
            tip_logits = clip_logits + cache_logits * tip_alpha
            tip_probs = torch.softmax(tip_logits, dim=1).squeeze()

            if clip_probs.shape[1] != num_id_classes:
                _logger.error(f"Z_p.shape: {clip_probs.shape} != id: {num_id_classes}")
                raise AssertionError

            top_clip_prob, _ = clip_probs.cpu().topk(1, dim=-1)
            clip_probs_max.extend(top_clip_prob.detach().numpy())
            top_tip_prob, _ = tip_probs.cpu().topk(1, dim=-1)
            tip_probs_max.extend(top_tip_prob.detach().numpy())

        targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)

        clip_aucs.append(get_auroc_for_max_probs(targets, clip_probs_max))
        tip_aucs.append(get_auroc_for_max_probs(targets, tip_probs_max))

    clip_mean, clip_std = get_mean_std(clip_aucs)
    tip_mean, tip_std = get_mean_std(tip_aucs)
    metrics = {'clip': clip_mean,
               'clip_std': clip_std,
               'tip': tip_mean,
               'tip_std': tip_std}

    return metrics


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

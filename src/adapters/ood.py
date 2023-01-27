import logging

import torch
from adapters.tip_adapter import get_cache_logits, get_cache_model, create_tip_train_set, \
    load_hyperparams_from_training, search_hp
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config
from torch.utils.data import DataLoader
from zoc.baseline import sorted_zeroshot_weights, get_zeroshot_weight_dict, get_feature_weight_dict
from zoc.utils import get_mean_std, get_auroc_for_max_probs, get_split_specific_targets, get_ablation_splits

from src.adapters.tip_adapter import get_dataset_with_shorted_classes, get_dataset_features_from_dataset_with_split

_logger = logging.getLogger(__name__)


def extract_full_split_features(shorted_val_loader, clip_model, param):
    pass


def tip_hyperparam_ood_detector(dset,
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
    for label_idx, split in enumerate(ablation_splits):
        _logger.info(f"Split ({label_idx + 1} / {len(ablation_splits)} ")

        seen_labels = split[:num_id_classes]
        unseen_labels = split[num_id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {unseen_labels}")
        zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)
        zeroshot_weights = zeroshot_weights.to(torch.float32)

        # get the kshot train set
        tip_train_set = create_tip_train_set(dset, seen_labels, kshots)
        _logger.info(f"len train set: {len(tip_train_set)}. Should be: {len(tip_train_set.classes) * kshots} (max)")
        cache_keys, cache_values = get_cache_model(tip_train_set, clip_model, augment_epochs=augment_epochs)

        # get shorted val set for the
        tip_val_set = get_dataset_with_shorted_classes(dset, seen_labels, kshots, 'val')
        shorted_val_loader = DataLoader(tip_val_set, batch_size=512)
        # get features from the shorted val set
        val_features, val_labels, label_features, classes = get_dataset_features_from_dataset_with_split(
            shorted_val_loader,
            clip_model,
            'val')
        clip_weights_val_set = 100 * val_features @ label_features
        alpha, beta = search_hp(cache_keys, cache_values, val_features, val_labels, clip_weights_val_set)

        clip_probs_max, tip_probs_max = [], []

        for label_idx, semantic_label in enumerate(split):
            # get features
            image_features_for_label = feature_weight_dict[semantic_label]
            image_features_for_label = image_features_for_label.to(torch.float32)

            # calc the logits and softmax
            clip_logits = image_features_for_label @ zeroshot_weights.T
            clip_probs = torch.softmax(clip_logits, dim=-1).squeeze()

            # TIP ADAPTER
            affinity = image_features_for_label @ cache_keys
            cache_logits = get_cache_logits(affinity, cache_values, beta)
            tip_logits = clip_logits + cache_logits * alpha
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
               'tip_std': tip_std,
               'alpha': alpha,
               'beta': beta}

    return metrics


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
        _logger.info(f"Split ({split_idx + 1} / {len(ablation_splits)} ")

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

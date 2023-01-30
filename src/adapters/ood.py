import logging

import torch
from adapters.tip_adapter import get_cache_logits, get_cache_model, \
    create_tip_train_set, load_hyperparams_from_training, \
    search_hp, get_dataset_with_shorted_classes, \
    run_tip_adapter_finetuned, WeightAdapter, \
    load_adapter, get_dataset_features_from_dataset_with_split
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config
from zoc.baseline import sorted_zeroshot_weights, get_zeroshot_weight_dict, get_feature_weight_dict
from zoc.utils import get_mean_std, get_auroc_for_max_probs, get_split_specific_targets, get_ablation_splits

_logger = logging.getLogger(__name__)


def tip_hyperparam_ood_detector(dset,
                                clip_model,
                                clip_transform,
                                device,
                                id_classes_split,
                                runs,
                                kshots,
                                augment_epochs,
                                train_epochs=None,
                                learning_rate=None,
                                eps=None,
                                finetune_adapter=False):
    if finetune_adapter:
        assert train_epochs and learning_rate and eps, "Missing params for finetuning"

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
    clip_aucs, tip_aucs, tipf_aucs = [], [], []
    for label_idx, split in enumerate(ablation_splits):
        _logger.info(f"Split ({label_idx + 1} / {len(ablation_splits)} ")

        seen_labels = split[:num_id_classes]
        unseen_labels = split[num_id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {unseen_labels}")
        zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)
        zeroshot_weights = zeroshot_weights.to(torch.float32)

        # get the kshot train set
        tip_train_set = create_tip_train_set(dset, seen_labels, kshots)
        tip_train_set.name = f"{tip_train_set.name}_{runs}_runs_ood"
        _logger.info(f"len train set: {len(tip_train_set)}. Should be: {len(tip_train_set.classes) * kshots} (max)")
        cache_keys, cache_values = get_cache_model(tip_train_set, clip_model, augment_epochs=augment_epochs)

        # get shorted val set for the
        tip_val_set = get_dataset_with_shorted_classes(dset, seen_labels, 'val')
        # get features from the shorted val set
        val_features, val_labels, label_features, classes = get_dataset_features_from_dataset_with_split(
            tip_val_set,
            clip_model)

        if finetune_adapter:
            # init alpha and beta according to paper

            # set init residual ratio to 1 ( new & old knowledge balanced)
            init_alpha = 1.
            # set sharpness nearly balanced
            init_beta = 1.17
            tipf_alpha, tipf_beta = run_tip_adapter_finetuned(tip_train_set, clip_model,
                                                              val_features, val_labels,
                                                              zeroshot_weights, cache_keys,
                                                              cache_values, init_alpha, init_beta,
                                                              train_epochs, learning_rate,
                                                              eps)
            tipf_adapter = WeightAdapter(cache_keys).to(device)
            tipf_adapter.load_state_dict(load_adapter(tip_train_set.name))
            tipf_adapter.eval()
        else:
            tip_alpha, tip_beta = search_hp(cache_keys, cache_values, val_features, val_labels, zeroshot_weights)

        clip_probs_max, tip_probs_max, tipf_probs_max = [], [], []

        for label_idx, semantic_label in enumerate(split):
            # get features
            test_image_features_for_label = feature_weight_dict[semantic_label]
            test_image_features_for_label = test_image_features_for_label.to(torch.float32)

            # calc the logits and softmax
            clip_logits = 100 * test_image_features_for_label @ zeroshot_weights.T
            clip_probs = torch.softmax(clip_logits, dim=-1).squeeze()
            top_clip_prob, _ = clip_probs.cpu().topk(1, dim=-1)
            clip_probs_max.extend(top_clip_prob.detach().numpy())

            if clip_probs.shape[1] != num_id_classes:
                _logger.error(f"Z_p.shape: {clip_probs.shape} != id: {num_id_classes}")
                raise AssertionError

            # TIP ADAPTER
            if finetune_adapter:
                tipf_adapter.eval()
                tipf_affinity = tipf_adapter(test_image_features_for_label)
                tipf_cache_logits = get_cache_logits(tipf_affinity, cache_values, tipf_beta)
                tipf_logits = clip_logits + tipf_cache_logits * tipf_alpha
                tipf_probs = torch.softmax(tipf_logits, dim=1).squeeze()
                top_tipf_prob, _ = tipf_probs.cpu().topk(1, dim=-1)
                tipf_probs_max.extend(top_tipf_prob.detach().numpy())

            tip_affinity = test_image_features_for_label @ cache_keys
            tip_cache_logits = get_cache_logits(tip_affinity, cache_values, tip_beta)
            tip_logits = clip_logits + tip_cache_logits * tip_alpha
            tip_probs = torch.softmax(tip_logits, dim=1).squeeze()
            top_tip_prob, _ = tip_probs.cpu().topk(1, dim=-1)
            tip_probs_max.extend(top_tip_prob.detach().numpy())

        targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)

        clip_aucs.append(get_auroc_for_max_probs(targets, clip_probs_max))
        tip_aucs.append(get_auroc_for_max_probs(targets, tip_probs_max))

    clip_mean, clip_std = get_mean_std(clip_aucs)
    tip_mean, tip_std = get_mean_std(tip_aucs)
    if finetune_adapter:
        tipf_aucs.append(get_auroc_for_max_probs(targets, tipf_probs_max))
        tipf_mean, tipf_std = get_mean_std(tipf_aucs)

        metrics = {'clip': clip_mean,
                   'clip_std': clip_std,
                   'tip': tip_mean,
                   'tip_std': tip_std,
                   'tip_alpha': tip_alpha,
                   'tip_beta': tip_beta,
                   'tipf': tipf_mean,
                   'tipf_std': tipf_std,
                   'tipf_alpha': tipf_alpha,
                   'tipf_beta': tipf_beta
                   }
    else:
        metrics = {'clip': clip_mean,
                   'clip_std': clip_std,
                   'tip': tip_mean,
                   'tip_std': tip_std,
                   'tip_alpha': tip_alpha,
                   'tip_beta': tip_beta}
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


def adapter_zoc():
    pass

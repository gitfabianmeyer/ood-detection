import logging

import numpy as np
import torch
import wandb
from adapters.oodd import get_ablation_split_classes, get_cosine_similarity_matrix_for_normed_features, \
    pad_list_of_vectors
from adapters.tip_adapter import create_tip_train_set, get_cache_model, get_dataset_with_shorted_classes, \
    get_dataset_features_from_dataset_with_split, run_tip_adapter_finetuned, search_hp, get_cache_logits
from datasets.zoc_loader import IsolatedClasses

from ood_detection.config import Config
from ood_detection.ood_utils import sorted_zeroshot_weights
from ood_detection.baseline import get_trained_linear_classifier

from zeroshot.utils import get_normalized_image_features, FeatureDict

from zoc.utils import get_ablation_splits, get_split_specific_targets, get_auroc_for_max_probs, get_mean_std, \
    get_zoc_unique_entities, tokenize_for_clip, get_auroc_for_ood_probs, get_caption_features_from_image_features, \
    get_zoc_feature_dict

from zeroshot.utils import get_zeroshot_weight_dict_from_isolated_classes

_logger = logging.getLogger(__name__)


def linear_adapter_zoc_ablation(dset,
                                clip_model,
                                clip_transform,
                                device,
                                id_classes_split,
                                augment_epochs,
                                runs_per_setting,
                                kshots,
                                train_epochs,
                                learning_rate,
                                eps,
                                shorten_classes=None):
    dataset = dset(data_path=Config.DATAPATH,
                   split='test',
                   transform=clip_transform)
    # prepare features ...
    isolated_classes_fast_loader = IsolatedClasses(dataset,
                                                   batch_size=512,
                                                   lsun=False)

    # CAREFUL: ADJUSTMENT FOR ZOC: THE TEMPLATES ( train tip on same )
    isolated_classes_fast_loader.templates = ["This is a photo of a {}"]
    _logger.info('Creating the test weight dicts')
    feature_weight_dict = FeatureDict(dataset, clip_model)
    classes_weight_dict = get_zeroshot_weight_dict_from_isolated_classes(isolated_classes_fast_loader, clip_model)

    _logger.info("Done creating weight dicts.")

    # prepare ablation splits...
    num_id_classes = int(len(dataset.classes) * id_classes_split)
    num_ood_classes = len(dataset.classes) - num_id_classes
    if shorten_classes:
        _logger.warning(f"SHORTENING CLASSES TO {shorten_classes}")
        num_id_classes = int(shorten_classes * Config.ID_SPLIT)
        num_ood_classes = shorten_classes - num_id_classes
    _logger.info(f"ID classes: {num_id_classes}, OOD classes: {num_ood_classes}")

    for kshot in kshots:

        ablation_splits = get_ablation_splits(dataset.classes, runs_per_setting, num_id_classes, num_ood_classes)

        # run for the ablation splits
        clip_aucs, tip_aucs, tipf_aucs, lin_aucs = [], [], [], []

        for split_idx, split in enumerate(ablation_splits):
            _logger.info(f"Split ({split_idx + 1} / {len(ablation_splits)} )")

            seen_descriptions, seen_labels, unseen_labels = get_ablation_split_classes(num_id_classes, split)

            # prep everything for tip(f)
            zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)
            zeroshot_weights = zeroshot_weights.to(torch.float32).to(device)

            # get the kshot train set
            tip_train_set = create_tip_train_set(dataset, seen_labels, kshot)
            tip_train_set.name = f"{tip_train_set.name}_ablation_kshot"
            _logger.info(f"len train set: {len(tip_train_set)}. Should be: {len(tip_train_set.classes) * kshot} (max)")
            cache_keys, cache_values = get_cache_model(tip_train_set, clip_model, augment_epochs=augment_epochs)

            # get shorted val set for the
            tip_val_set = get_dataset_with_shorted_classes(dset, seen_labels, 'val')
            # get features from the shorted val set
            val_features, val_labels, label_features, classes = get_dataset_features_from_dataset_with_split(
                tip_val_set,
                clip_model)

            # linear stuff
            linear_classifier = get_trained_linear_classifier(tip_train_set, tip_val_set, seen_labels, clip_model,
                                                              device)

            tipf_alpha, tipf_beta, tipf_adapter = run_tip_adapter_finetuned(tip_train_set, clip_model,
                                                                            val_features, val_labels,
                                                                            zeroshot_weights, cache_keys,
                                                                            cache_values, train_epochs,
                                                                            learning_rate, eps)

            tip_alpha, tip_beta = search_hp(cache_keys, cache_values, val_features, val_labels, zeroshot_weights)
            # run OOD
            clip_probs_max, tip_probs_max, tipf_probs_max, linear_probs_max = [], [], [], []

            for label_idx, semantic_label in enumerate(split):

                # get features
                test_image_features_for_label = feature_weight_dict[semantic_label]
                test_image_features_for_label = test_image_features_for_label.to(torch.float32)

                # calc the logits and softmax
                clip_logits = get_cosine_similarity_matrix_for_normed_features(test_image_features_for_label,
                                                                               zeroshot_weights, 0.01)
                clip_probs = torch.softmax(clip_logits, dim=-1).squeeze()
                top_clip_prob, _ = clip_probs.cpu().topk(1, dim=-1)
                clip_probs_max.extend(top_clip_prob.detach().numpy())

                if clip_probs.shape[1] != num_id_classes:
                    _logger.error(f"Z_p.shape: {clip_probs.shape} != id: {num_id_classes}")
                    raise AssertionError

                # linear
                with torch.no_grad():
                    linear_logits = linear_classifier(test_image_features_for_label)
                top_linear_prob, _ = linear_logits.cpu().topk(1, dim=-1)
                linear_probs_max.extend(top_linear_prob.detach().numpy())

                # TIPF ADAPTER
                tipf_affinity = tipf_adapter(test_image_features_for_label)
                tipf_cache_logits = get_cache_logits(tipf_affinity, cache_values, tipf_beta)
                tipf_logits = clip_logits + tipf_cache_logits * tipf_alpha
                tipf_probs = torch.softmax(tipf_logits, dim=1).squeeze()
                top_tipf_prob, _ = tipf_probs.cpu().topk(1, dim=-1)
                tipf_probs_max.extend(top_tipf_prob.detach().numpy())

                # tip
                tip_affinity = test_image_features_for_label @ cache_keys
                tip_cache_logits = get_cache_logits(tip_affinity, cache_values, tip_beta)
                tip_logits = clip_logits + tip_cache_logits * tip_alpha
                tip_probs = torch.softmax(tip_logits, dim=1).squeeze()
                top_tip_prob, _ = tip_probs.cpu().topk(1, dim=-1)
                tip_probs_max.extend(top_tip_prob.detach().numpy())

            targets = get_split_specific_targets(isolated_classes_fast_loader, seen_labels, unseen_labels)

            assert len(targets) == len(clip_probs_max), f"{len(targets)} != {len(clip_probs_max)}"
            assert len(targets) == len(tip_probs_max), f"{len(targets)} != {len(tip_probs_max)}"
            assert len(targets) == len(tipf_probs_max), f"{len(targets)} != {len(tipf_probs_max)}"

            clip_aucs.append(get_auroc_for_max_probs(targets, np.array(clip_probs_max)))
            tip_aucs.append(get_auroc_for_max_probs(targets, tip_probs_max))
            tipf_aucs.append(get_auroc_for_max_probs(targets, tipf_probs_max))
            lin_aucs.append(get_auroc_for_max_probs(targets, linear_probs_max))

        # summed up over splits
        clip_mean, clip_std = get_mean_std(clip_aucs)
        tip_mean, tip_std = get_mean_std(tip_aucs)
        tipf_mean, tipf_std = get_mean_std(tipf_aucs)
        linear_mean, linear_std = get_mean_std(lin_aucs)

        metrics = {'clip': clip_mean,
                   'clip_std': clip_std,
                   'tip': tip_mean,
                   'tip_std': tip_std,
                   'tipf': tipf_mean,
                   'tipf_std': tipf_std,
                   'linear': linear_mean,
                   'linear_std': linear_std,
                   'shots': kshot
                   }
        return metrics


def splits_adapter_zoc_ablation(dset,
                                clip_model,
                                clip_transform,
                                clip_tokenizer,
                                bert_tokenizer,
                                bert_model,
                                device,
                                id_classes_splits,
                                augment_epochs,
                                runs_per_setting,
                                kshots,
                                train_epochs,
                                learning_rate,
                                eps,
                                shorten_classes=None):
    dataset = dset(data_path=Config.DATAPATH,
                   split='test',
                   transform=clip_transform)
    # prepare features ...
    zoc_unique_entities = get_zoc_unique_entities(dataset, clip_model, bert_tokenizer, bert_model)

    isolated_classes_fast_loader = IsolatedClasses(dataset,
                                                   batch_size=512,
                                                   lsun=False)

    # CAREFUL: ADJUSTMENT FOR ZOC: THE TEMPLATES ( train tip on same )
    isolated_classes_fast_loader.templates = ["This is a photo of a {}"]
    _logger.info('Creating the test weight dicts')
    feature_weight_dict = FeatureDict(dataset, clip_model)
    classes_weight_dict = get_zeroshot_weight_dict_from_isolated_classes(isolated_classes_fast_loader, clip_model)
    _logger.info("Done creating weight dicts.")

    # prepare ablation splits...
    for id_classes_split in id_classes_splits:
        num_id_classes = int(len(dataset.classes) * id_classes_split)
        num_ood_classes = len(dataset.classes) - num_id_classes
        if shorten_classes:
            _logger.warning(f"SHORTENING CLASSES TO {shorten_classes}")
            num_id_classes = int(shorten_classes * Config.ID_SPLIT)
            num_ood_classes = shorten_classes - num_id_classes
        _logger.info(f"ID classes: {num_id_classes}, OOD classes: {num_ood_classes}")

        isolated_classes_slow_loader = IsolatedClasses(dataset,
                                                       batch_size=1,
                                                       lsun=False)
        ablation_splits = get_ablation_splits(dataset.classes, runs_per_setting, num_id_classes, num_ood_classes)

        # run for the ablation splits
        clip_aucs, tip_aucs, tipf_aucs = [], [], []
        zoc_aucs, toc_aucs, tocf_aucs = [], [], []

        for split_idx, split in enumerate(ablation_splits):
            _logger.info(f"Split ({split_idx + 1} / {len(ablation_splits)} )")

            seen_descriptions, seen_labels, unseen_labels = get_ablation_split_classes(num_id_classes, split)

            # prep everything for tip(f)
            zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)
            zeroshot_weights = zeroshot_weights.to(torch.float32)

            # get the kshot train set
            tip_train_set = create_tip_train_set(dataset, seen_labels, kshots)
            tip_train_set.name = f"{tip_train_set.name}_ablation_splits"
            _logger.info(f"len train set: {len(tip_train_set)}. Should be: {len(tip_train_set.classes) * kshots} (max)")
            cache_keys, cache_values = get_cache_model(tip_train_set, clip_model, augment_epochs=augment_epochs)

            # get shorted val set for the
            tip_val_set = get_dataset_with_shorted_classes(dset, seen_labels, 'val')
            # get features from the shorted val set
            val_features, val_labels, label_features, classes = get_dataset_features_from_dataset_with_split(
                tip_val_set,
                clip_model)

            tipf_alpha, tipf_beta, tipf_adapter = run_tip_adapter_finetuned(tip_train_set, clip_model,
                                                                            val_features, val_labels,
                                                                            zeroshot_weights, cache_keys,
                                                                            cache_values, train_epochs,
                                                                            learning_rate, eps)

            tip_alpha, tip_beta = search_hp(cache_keys, cache_values, val_features, val_labels, zeroshot_weights)
            # run zoc
            clip_probs_max, tip_probs_max, tipf_probs_max = [], [], []
            zoc_probs_sum, toc_probs_sum, tocf_probs_sum = [], [], [],

            for label_idx, semantic_label in enumerate(split):

                # get features
                test_image_features_for_label = feature_weight_dict[semantic_label]
                test_image_features_for_label = test_image_features_for_label.to(torch.float32)

                # calc the logits and softmax
                clip_logits = get_cosine_similarity_matrix_for_normed_features(test_image_features_for_label,
                                                                               zeroshot_weights, 0.01)
                clip_probs = torch.softmax(clip_logits, dim=-1).squeeze()
                top_clip_prob, _ = clip_probs.cpu().topk(1, dim=-1)
                clip_probs_max.extend(top_clip_prob.detach().numpy())

                if clip_probs.shape[1] != num_id_classes:
                    _logger.error(f"Z_p.shape: {clip_probs.shape} != id: {num_id_classes}")
                    raise AssertionError

                # ZOC
                zoc_entities_for_semantic_label = zoc_unique_entities[semantic_label]

                zoc_logits_for_semantic_label = []

                loader = isolated_classes_slow_loader[semantic_label]
                for image, unique_entities in zip(loader, zoc_entities_for_semantic_label):
                    all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
                    all_desc_ids = tokenize_for_clip(all_desc, clip_tokenizer)
                    with torch.no_grad():
                        image_feature = clip_model.encode_image(image.to(device)).float()
                        image_feature /= image_feature.norm(dim=-1, keepdim=True)
                        image_feature = image_feature.to(torch.float32)
                        text_features = clip_model.encode_text(all_desc_ids.to(device)).to(torch.float32)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                    zoc_logits_for_image = get_cosine_similarity_matrix_for_normed_features(image_feature,
                                                                                            text_features, 0.01)
                    zoc_logits_for_semantic_label.append(zoc_logits_for_image)
                    zoc_probs = torch.softmax(zoc_logits_for_image, dim=-1)
                    zoc_probs_sum.append(torch.sum(zoc_probs[len(seen_labels):]))  # for normal zoc

                # now: use normal zoc probs. use zoctip. use zoctipf

                # first, pad all to then longest with -inf (neutral element in softmax)
                padded_zoc_logits_for_semantic_label = pad_list_of_vectors(zoc_logits_for_semantic_label, -np.inf)

                # TIPF ADAPTER
                tipf_affinity = tipf_adapter(test_image_features_for_label)
                tipf_cache_logits = get_cache_logits(tipf_affinity, cache_values, tipf_beta)
                tipf_logits = clip_logits + tipf_cache_logits * tipf_alpha
                tipf_probs = torch.softmax(tipf_logits, dim=1).squeeze()
                top_tipf_prob, _ = tipf_probs.cpu().topk(1, dim=-1)
                tipf_probs_max.extend(top_tipf_prob.detach().numpy())

                # tip
                tip_affinity = test_image_features_for_label @ cache_keys

                tip_cache_logits = get_cache_logits(tip_affinity, cache_values, tip_beta)
                tip_logits = clip_logits + tip_cache_logits * tip_alpha
                tip_probs = torch.softmax(tip_logits, dim=1).squeeze()
                top_tip_prob, _ = tip_probs.cpu().topk(1, dim=-1)
                tip_probs_max.extend(top_tip_prob.detach().numpy())

                # zoc tip
                padded_cache_logits = torch.zeros(padded_zoc_logits_for_semantic_label.shape)
                padded_cache_logits[:, :tip_cache_logits.shape[1]] = tip_cache_logits
                # the magic
                toc_logits = padded_zoc_logits_for_semantic_label + padded_cache_logits * tip_alpha
                toc_probs = torch.softmax(toc_logits, dim=1).squeeze()
                toc_probs_sum.extend(torch.sum(toc_probs[:, len(seen_labels):], dim=1).detach().numpy())

                # zoc tipf
                padded_cache_logits = torch.zeros(padded_zoc_logits_for_semantic_label.shape)
                padded_cache_logits[:, :tipf_cache_logits.shape[1]] = tipf_cache_logits
                # the magic
                tocf_logits = padded_zoc_logits_for_semantic_label + padded_cache_logits * tipf_alpha
                tocf_probs = torch.softmax(tocf_logits, dim=1).squeeze()
                tocf_probs_sum.extend(torch.sum(tocf_probs[:, len(seen_labels):], dim=1).detach().numpy())

            targets = get_split_specific_targets(isolated_classes_fast_loader, seen_labels, unseen_labels)

            assert len(targets) == len(zoc_probs_sum), f"{len(targets)} != {len(zoc_probs_sum)}"
            assert len(targets) == len(clip_probs_max), f"{len(targets)} != {len(clip_probs_max)}"
            assert len(targets) == len(toc_probs_sum), f"{len(targets)} != {len(tocf_probs_sum)}"
            assert len(targets) == len(tocf_probs_sum), f"{len(targets)} != {len(tocf_probs_sum)}"
            assert len(targets) == len(tip_probs_max), f"{len(targets)} != {len(tip_probs_max)}"
            assert len(targets) == len(tipf_probs_max), f"{len(targets)} != {len(tipf_probs_max)}"

            clip_aucs.append(get_auroc_for_max_probs(targets, np.array(clip_probs_max)))
            tip_aucs.append(get_auroc_for_max_probs(targets, tip_probs_max))
            tipf_aucs.append(get_auroc_for_max_probs(targets, tipf_probs_max))
            zoc_aucs.append(get_auroc_for_ood_probs(targets, zoc_probs_sum))
            toc_aucs.append(get_auroc_for_ood_probs(targets, toc_probs_sum))
            tocf_aucs.append(get_auroc_for_ood_probs(targets, tocf_probs_sum))

        # summed up over splits
        clip_mean, clip_std = get_mean_std(clip_aucs)
        tip_mean, tip_std = get_mean_std(tip_aucs)
        tipf_mean, tipf_std = get_mean_std(tipf_aucs)

        zoc_mean, zoc_std = get_mean_std(zoc_aucs)
        toc_mean, toc_std = get_mean_std(toc_aucs)
        tocf_mean, tocf_std = get_mean_std(tocf_aucs)

        metrics = {'clip': clip_mean,
                   'clip_std': clip_std,
                   'tip': tip_mean,
                   'tip_std': tip_std,
                   'tipf': tipf_mean,
                   'tipf_std': tipf_std,
                   'zoc': zoc_mean,
                   'zoc_std': zoc_std,
                   'toc': toc_mean,
                   'toc_std': toc_std,
                   'tocf': tocf_mean,
                   'tocf_std': tocf_std,
                   'seen_labels': num_id_classes
                   }
        return metrics


def zoc_temp_ablation(dset,
                      clip_model,
                      clip_transform,
                      runs_per_setting,
                      temperatures):
    dataset = dset(data_path=Config.DATAPATH,
                   split='test',
                   transform=clip_transform)

    isolated_classes_fast_loader = IsolatedClasses(dataset,
                                                   batch_size=512,
                                                   lsun=False)

    isolated_classes_fast_loader.templates = ["This is a photo of a {}"]
    _logger.info('Creating the test weight dicts')
    feature_weight_dict = FeatureDict(dataset, clip_model)

    num_id_classes = int(len(dataset.classes) * Config.ID_SPLIT)
    num_ood_classes = len(dataset.classes) - num_id_classes

    ablation_splits = get_ablation_splits(dataset.classes, runs_per_setting, num_id_classes, num_ood_classes)

    for temperature in temperatures:
        zoc_aucs = []
        for ablation_split in ablation_splits:

            seen_descriptions, seen_labels, unseen_labels = get_ablation_split_classes(num_id_classes, ablation_split)
            zoc_featuredict = get_zoc_feature_dict(dataset, clip_model, seen_labels)

            zoc_probs_sum = []
            for semantic_label in ablation_split:

                image_features = feature_weight_dict[semantic_label]
                image_features = image_features.to(torch.float32)

                zoc_label_features = zoc_featuredict[semantic_label]

                for image_feature, zoc_label_feature in zip(image_features, zoc_label_features):
                    zoc_label_feature = zoc_label_feature.to(torch.float32)
                    similarity = get_cosine_similarity_matrix_for_normed_features(image_feature,
                                                                                  zoc_label_feature,
                                                                                  temperature)
                    id_similarity = torch.sum(
                        torch.softmax(similarity, dim=-1)[num_id_classes:]
                    )
                    zoc_probs_sum.append(id_similarity.cpu())

            targets = get_split_specific_targets(isolated_classes_fast_loader, seen_labels, unseen_labels)
            assert len(targets) == len(zoc_probs_sum), f"{len(targets)} != {len(zoc_probs_sum)}"
            zoc_aucs.append(get_auroc_for_ood_probs(targets, zoc_probs_sum))

        zoc_mean, zoc_std = get_mean_std(zoc_aucs)

        metrics = {
            'temperature': temperature,
            'zoc': zoc_mean,
            'zoc_std': zoc_std}
        wandb.log(metrics)
    return True


def kshot_adapter_zoc_ablation(dset,
                               clip_model,
                               clip_transform,
                               clip_tokenizer,
                               bert_tokenizer,
                               bert_model,
                               id_classes_split,
                               augment_epochs,
                               runs_per_setting,
                               kshots,
                               train_epochs,
                               learning_rate,
                               eps,
                               shorten_classes=None):
    dataset = dset(data_path=Config.DATAPATH,
                   split='test',
                   transform=clip_transform)
    # prepare features ...
    isolated_classes_fast_loader = IsolatedClasses(dataset,
                                                   batch_size=512,
                                                   lsun=False)

    # CAREFUL: ADJUSTMENT FOR ZOC: THE TEMPLATES ( train tip on same )
    isolated_classes_fast_loader.templates = ["This is a photo of a {}"]
    _logger.info('Creating the test weight dicts')
    feature_weight_dict = FeatureDict(dataset, clip_model)
    classes_weight_dict = get_zeroshot_weight_dict_from_isolated_classes(isolated_classes_fast_loader, clip_model)
    _logger.info("Done creating weight dicts.")

    # prepare ablation splits...
    num_id_classes = int(len(dataset.classes) * id_classes_split)
    num_ood_classes = len(dataset.classes) - num_id_classes
    if shorten_classes:
        _logger.warning(f"SHORTENING CLASSES TO {shorten_classes}")
        num_id_classes = int(shorten_classes * Config.ID_SPLIT)
        num_ood_classes = shorten_classes - num_id_classes
    _logger.info(f"ID classes: {num_id_classes}, OOD classes: {num_ood_classes}")

    all_seen_descriptions = [f"This is a photo of a {label}" for label in dataset.classes]
    zoc_unique_entities = get_zoc_unique_entities(dataset, all_seen_descriptions, clip_model, clip_tokenizer,
                                                  bert_tokenizer,
                                                  bert_model)

    isolated_classes_slow_loader = IsolatedClasses(dataset,
                                                   batch_size=1,
                                                   lsun=False)

    for kshot in kshots:

        ablation_splits = get_ablation_splits(dataset.classes, runs_per_setting, num_id_classes, num_ood_classes)

        # run for the ablation splits
        clip_aucs, tip_aucs, tipf_aucs = [], [], []
        zoc_aucs, toc_aucs, tocf_aucs = [], [], []

        for split_idx, split in enumerate(ablation_splits):
            _logger.info(f"Split ({split_idx + 1} / {len(ablation_splits)} )")

            seen_descriptions, seen_labels, unseen_labels = get_ablation_split_classes(num_id_classes, split)

            # prep everything for tip(f)
            zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)
            zeroshot_weights = zeroshot_weights.to(torch.float32)

            # get the kshot train set
            tip_train_set = create_tip_train_set(dataset, seen_labels, kshot)
            tip_train_set.name = f"{tip_train_set.name}_ablation_kshot"
            _logger.info(f"len train set: {len(tip_train_set)}. Should be: {len(tip_train_set.classes) * kshots} (max)")
            cache_keys, cache_values = get_cache_model(tip_train_set, clip_model, augment_epochs=augment_epochs)

            # get shorted val set for the
            tip_val_set = get_dataset_with_shorted_classes(dset, seen_labels, 'val')
            # get features from the shorted val set
            val_features, val_labels, label_features, classes = get_dataset_features_from_dataset_with_split(
                tip_val_set,
                clip_model)

            # set init residual ratio to 1 ( new & old knowledge balanced)
            init_alpha = 1.
            # set sharpness nearly balanced
            init_beta = 1.17
            tipf_alpha, tipf_beta, tipf_adapter = run_tip_adapter_finetuned(tip_train_set, clip_model,
                                                                            val_features, val_labels,
                                                                            zeroshot_weights, cache_keys,
                                                                            cache_values, init_alpha, init_beta,
                                                                            train_epochs, learning_rate,
                                                                            eps)

            tip_alpha, tip_beta = search_hp(cache_keys, cache_values, val_features, val_labels, zeroshot_weights)
            # run zoc
            clip_probs_max, tip_probs_max, tipf_probs_max = [], [], []
            zoc_probs_sum, toc_probs_sum, tocf_probs_sum = [], [], [],

            for label_idx, semantic_label in enumerate(split):

                # get features
                test_image_features_for_label = feature_weight_dict[semantic_label]
                test_image_features_for_label = test_image_features_for_label.to(torch.float32)

                # calc the logits and softmax
                clip_logits = get_cosine_similarity_matrix_for_normed_features(test_image_features_for_label,
                                                                               zeroshot_weights, 0.01)
                clip_probs = torch.softmax(clip_logits, dim=-1).squeeze()
                top_clip_prob, _ = clip_probs.cpu().topk(1, dim=-1)
                clip_probs_max.extend(top_clip_prob.detach().numpy())

                if clip_probs.shape[1] != num_id_classes:
                    _logger.error(f"Z_p.shape: {clip_probs.shape} != id: {num_id_classes}")
                    raise AssertionError

                # ZOC
                zoc_entities_for_semantic_label = zoc_unique_entities[semantic_label]

                zoc_logits_for_semantic_label = []

                loader = isolated_classes_slow_loader[semantic_label]
                for image, unique_entities in zip(loader, zoc_entities_for_semantic_label):
                    text_features = get_caption_features_from_image_features(image, seen_descriptions,
                                                                             seen_labels, bert_model,
                                                                             bert_tokenizer, clip_model,
                                                                             clip_tokenizer)
                    image_feature = get_normalized_image_features(clip_model, image)

                    zoc_logits_for_image = get_cosine_similarity_matrix_for_normed_features(image_feature,
                                                                                            text_features, 0.01)
                    zoc_logits_for_semantic_label.append(zoc_logits_for_image)
                    zoc_probs = torch.softmax(zoc_logits_for_image, dim=-1)
                    zoc_probs_sum.append(torch.sum(zoc_probs[len(seen_labels):]))  # for normal zoc

                # now: use normal zoc probs. use zoctip. use zoctipf

                # first, pad all to then longest with -inf (neutral element in softmax)
                padded_zoc_logits_for_semantic_label = pad_list_of_vectors(zoc_logits_for_semantic_label, -np.inf)

                # TIPF ADAPTER
                tipf_affinity = tipf_adapter(test_image_features_for_label)
                tipf_cache_logits = get_cache_logits(tipf_affinity, cache_values, tipf_beta)
                tipf_logits = clip_logits + tipf_cache_logits * tipf_alpha
                tipf_probs = torch.softmax(tipf_logits, dim=1).squeeze()
                top_tipf_prob, _ = tipf_probs.cpu().topk(1, dim=-1)
                tipf_probs_max.extend(top_tipf_prob.detach().numpy())

                # tip
                tip_affinity = test_image_features_for_label @ cache_keys
                tip_cache_logits = get_cache_logits(tip_affinity, cache_values, tip_beta)
                tip_logits = clip_logits + tip_cache_logits * tip_alpha
                tip_probs = torch.softmax(tip_logits, dim=1).squeeze()
                top_tip_prob, _ = tip_probs.cpu().topk(1, dim=-1)
                tip_probs_max.extend(top_tip_prob.detach().numpy())

                # zoc tip
                padded_cache_logits = torch.zeros(padded_zoc_logits_for_semantic_label.shape)
                padded_cache_logits[:, :tip_cache_logits.shape[1]] = tip_cache_logits
                # the magic
                toc_logits = padded_zoc_logits_for_semantic_label + padded_cache_logits * tip_alpha
                toc_probs = torch.softmax(toc_logits, dim=1).squeeze()
                toc_probs_sum.extend(torch.sum(toc_probs[:, len(seen_labels):], dim=1).detach().numpy())

                # zoc tipf
                padded_cache_logits = torch.zeros(padded_zoc_logits_for_semantic_label.shape)
                padded_cache_logits[:, :tipf_cache_logits.shape[1]] = tipf_cache_logits
                # the magic
                tocf_logits = padded_zoc_logits_for_semantic_label + padded_cache_logits * tipf_alpha
                tocf_probs = torch.softmax(tocf_logits, dim=1).squeeze()
                tocf_probs_sum.extend(torch.sum(tocf_probs[:, len(seen_labels):], dim=1).detach().numpy())

            targets = get_split_specific_targets(isolated_classes_fast_loader, seen_labels, unseen_labels)

            assert len(targets) == len(zoc_probs_sum), f"{len(targets)} != {len(zoc_probs_sum)}"
            assert len(targets) == len(clip_probs_max), f"{len(targets)} != {len(clip_probs_max)}"
            assert len(targets) == len(toc_probs_sum), f"{len(targets)} != {len(tocf_probs_sum)}"
            assert len(targets) == len(tocf_probs_sum), f"{len(targets)} != {len(tocf_probs_sum)}"
            assert len(targets) == len(tip_probs_max), f"{len(targets)} != {len(tip_probs_max)}"
            assert len(targets) == len(tipf_probs_max), f"{len(targets)} != {len(tipf_probs_max)}"

            clip_aucs.append(get_auroc_for_max_probs(targets, np.array(clip_probs_max)))
            tip_aucs.append(get_auroc_for_max_probs(targets, tip_probs_max))
            tipf_aucs.append(get_auroc_for_max_probs(targets, tipf_probs_max))
            zoc_aucs.append(get_auroc_for_ood_probs(targets, zoc_probs_sum))
            toc_aucs.append(get_auroc_for_ood_probs(targets, toc_probs_sum))
            tocf_aucs.append(get_auroc_for_ood_probs(targets, tocf_probs_sum))

        # summed up over splits
        clip_mean, clip_std = get_mean_std(clip_aucs)
        tip_mean, tip_std = get_mean_std(tip_aucs)
        tipf_mean, tipf_std = get_mean_std(tipf_aucs)

        zoc_mean, zoc_std = get_mean_std(zoc_aucs)
        toc_mean, toc_std = get_mean_std(toc_aucs)
        tocf_mean, tocf_std = get_mean_std(tocf_aucs)

        metrics = {'clip': clip_mean,
                   'clip_std': clip_std,
                   'tip': tip_mean,
                   'tip_std': tip_std,
                   'tipf': tipf_mean,
                   'tipf_std': tipf_std,
                   'zoc': zoc_mean,
                   'zoc_std': zoc_std,
                   'toc': toc_mean,
                   'toc_std': toc_std,
                   'tocf': tocf_mean,
                   'tocf_std': tocf_std,
                   'shots': kshot
                   }
        return metrics


@torch.no_grad()
def get_clip_auroc_from_features(id_features, ood_features, zeroshot_weights, temperature, strategy):
    assert strategy in ["msp", "mlp"]
    top_probs = []
    for features in [id_features, ood_features]:
        zsw = get_cosine_similarity_matrix_for_normed_features(features, zeroshot_weights, temperature)
        if strategy == 'msp':
            clip_probs = torch.softmax(zsw, dim=-1).squeeze()
        top_clip_prob, _ = clip_probs.cpu().topk(1, dim=-1)
        top_probs.extend(top_clip_prob)

    top_probs = torch.stack(top_probs).squeeze()
    targets = torch.Tensor([0] * len(id_features) + [1] * len(ood_features))
    score = get_auroc_for_max_probs(targets, top_probs)
    return score

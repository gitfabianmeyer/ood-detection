import logging

import numpy as np
import torch
from adapters.tip_adapter import get_cache_logits, get_cache_model, \
    create_tip_train_set, load_hyperparams_from_training, \
    search_hp, get_dataset_with_shorted_classes, \
    run_tip_adapter_finetuned, WeightAdapter, \
    load_adapter, get_dataset_features_from_dataset_with_split
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config
from tqdm import tqdm
from zoc.baseline import sorted_zeroshot_weights, get_zeroshot_weight_dict, get_feature_weight_dict
from zoc.utils import get_mean_std, get_auroc_for_max_probs, get_split_specific_targets, get_ablation_splits, \
    greedysearch_generation_topk, tokenize_for_clip, get_auroc_for_ood_probs

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


def pad_list_of_vectors(list_of_vectors, value=-np.inf, max_length=None):
    new_vectors = []
    if not max_length:
        max_length = len(max(list_of_vectors, key=len))
    for vector in list_of_vectors:
        if len(vector) < max_length:
            zeros = torch.zeros((max_length,))
            zeros[zeros == 0] = value
            zeros[:len(vector)] = vector
            new_vectors.append(zeros)
        else:
            new_vectors.append(vector)
    return torch.stack(new_vectors)


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


# 1. train tip(f) on ablation split to provide extra knowledge to known classes.
# 2. Run zoc
# 3. Add Knowledge to known classes
# 4. Get the aurocs
def adapter_zoc(dset,
                clip_model,
                clip_transform,
                clip_tokenizer,
                bert_tokenizer,
                bert_model,
                device,
                id_classes_split,
                augment_epochs,
                runs,
                kshots,
                train_epochs,
                learning_rate,
                eps):
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
    feature_weight_dict = get_feature_weight_dict(isolated_classes_fast_loader, clip_model, device)
    classes_weight_dict = get_zeroshot_weight_dict(isolated_classes_fast_loader, clip_model)
    _logger.info("Done creating weight dicts.")

    # prepare ablation splits...
    num_id_classes = int(len(dataset.classes) * id_classes_split)
    num_ood_classes = len(dataset.classes) - num_id_classes
    _logger.info(f"ID classes: {num_id_classes}, OOD classes: {num_ood_classes}")
    ablation_splits = get_ablation_splits(dataset.classes, runs, num_id_classes, num_ood_classes)

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
        tip_train_set = create_tip_train_set(dset, seen_labels, kshots)
        tip_train_set.name = f"{tip_train_set.name}_{runs}-runs_ood_split-{split_idx}"
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
        tipf_alpha, tipf_beta = run_tip_adapter_finetuned(tip_train_set, clip_model,
                                                          val_features, val_labels,
                                                          zeroshot_weights, cache_keys,
                                                          cache_values, init_alpha, init_beta,
                                                          train_epochs, learning_rate,
                                                          eps)
        tipf_adapter = WeightAdapter(cache_keys).to(device)
        tipf_adapter.load_state_dict(load_adapter(tip_train_set.name))
        tipf_adapter.eval()

        tip_alpha, tip_beta = search_hp(cache_keys, cache_values, val_features, val_labels, zeroshot_weights)
        # run zoc
        clip_probs_max, tip_probs_max, tipf_probs_max = [], [], []
        zoc_probs_sum, toc_probs_sum, tocf_probs_sum = [], [], [],

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

            # ZOC
            zoc_logits_for_semantic_label = []
            isolated_classes_slow_loader = IsolatedClasses(dataset,
                                                           batch_size=1,
                                                           lsun=False)
            loader = isolated_classes_slow_loader[semantic_label]

            for image_idx, image in enumerate(tqdm(loader)):
                with torch.no_grad():

                    clip_out = clip_model.encode_image(image.to(device)).float()
                    clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)

                    # greedy generation
                    target_list, topk_list = greedysearch_generation_topk(clip_extended_embed,
                                                                      bert_tokenizer,
                                                                      bert_model,
                                                                      device)

                    topk_tokens = [bert_tokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]

                unique_entities = list(set(topk_tokens) - {semantic_label})
                _logger.debug(f"Semantic label: {semantic_label}Unique Entities: {unique_entities}")

                all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
                all_desc_ids = tokenize_for_clip(all_desc, clip_tokenizer)

                image_feature = clip_out
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                image_feature = image_feature.to(torch.float32)

                with torch.no_grad():
                    text_features = clip_model.encode_text(all_desc_ids.to(device)).to(torch.float32)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                zoc_logits_for_image = (100.0 * image_feature @ text_features.T).squeeze().cpu()
                zoc_probs = torch.softmax(zoc_logits_for_image, dim=0)
                zoc_probs_sum.append(torch.sum(zoc_probs[len(seen_labels):]))  # for normal zoc
                zoc_logits_for_semantic_label.append(zoc_logits_for_image)  # for toc/f

            # now: use normal zoc probs. use zoctip. use zoctipf

            # first, pad all to then longest with -inf (neutral element in softmax)
            zoc_logits_for_semantic_label = pad_list_of_vectors(zoc_logits_for_semantic_label, -np.inf)

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
            padded_cache_logits = torch.zeros(zoc_logits_for_semantic_label.shape)
            padded_cache_logits[:, :tip_cache_logits.shape[1]] = tip_cache_logits
            #the magic
            toc_logits = zoc_logits_for_semantic_label + padded_cache_logits * tip_alpha
            toc_probs = torch.softmax(toc_logits, dim=1).squeeze()
            toc_probs_sum.extend(torch.sum(toc_probs[:, len(seen_labels):], dim=1).detach().numpy())

            # zoc tipf
            padded_cache_logits = torch.zeros(zoc_logits_for_semantic_label.shape)
            padded_cache_logits[:, :tipf_cache_logits.shape[1]] = tipf_cache_logits
            # the magic
            tocf_logits = zoc_logits_for_semantic_label + padded_cache_logits * tipf_alpha
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
               'tocf_std': tocf_std
               }
    return metrics


def get_ablation_split_classes(num_id_classes, split):
    seen_labels = split[:num_id_classes]
    seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]
    unseen_labels = split[num_id_classes:]
    _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {unseen_labels}")
    return seen_descriptions, seen_labels, unseen_labels

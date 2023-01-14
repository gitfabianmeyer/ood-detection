import logging
import random

import numpy as np
import torch
from adapters.tip_adapter import get_train_transform, get_kshot_train_set, get_adapter_weights
from ood_detection.config import Config
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.zoc_loader import IsolatedClasses
from adapters.tip_adapter import get_train_features, WeightAdapter

_logger = logging.getLogger()

@torch.no_grad()
def greedysearch_generation_topk(clip_embed, berttokenizer, bert_model, device):
    N = 1  # batch has single sample
    max_len = 77
    target_list = [torch.tensor(berttokenizer.bos_token_id)]
    top_k_list = []
    bert_model.eval()
    for _ in range(max_len):
        target = torch.LongTensor(target_list).unsqueeze(0)
        position_ids = torch.arange(0, len(target)).expand(N, len(target)).to(device)
        out = bert_model(input_ids=target.to(device),
                         position_ids=position_ids,
                         attention_mask=torch.ones(len(target)).unsqueeze(0).to(device),
                         encoder_hidden_states=clip_embed.unsqueeze(1).to(device),
                         )

        pred_idx = out.logits.argmax(2)[:, -1]
        _, top_k = torch.topk(out.logits, dim=2, k=35)
        top_k_list.append(top_k[:, -1].flatten())
        target_list.append(pred_idx)
        if len(target_list) == 10:  # the entitiy word is in at most first 10 words
            break
    top_k_list = torch.cat(top_k_list)
    return target_list, top_k_list


def tokenize_for_clip(batch_sentences, tokenizer):
    default_length = 77  # CLIP default
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    tokenized_list = []
    for sentence in batch_sentences:
        text_tokens = [sot_token] + tokenizer.encode(sentence) + [eot_token]
        tokenized = torch.zeros((default_length), dtype=torch.long)
        tokenized[:len(text_tokens)] = torch.tensor(text_tokens)
        tokenized_list.append(tokenized)
    tokenized_list = torch.stack(tokenized_list)
    return tokenized_list


def get_ablation_splits(classnames, n, id_classes, ood_classes=None):
    assert ood_classes, 'Missing ood classes when using int split'
    if id_classes + ood_classes > len(classnames):
        raise IndexError("Too few classes to build split")

    splits = []
    for _ in range(n):
        base = random.sample(classnames, k=id_classes)
        leftover = [classname for classname in classnames if classname not in base]
        oods = random.sample(leftover, k=ood_classes)
        splits.append(base + oods)
    for split in splits:
        assert len(split) == len(set(split))
    return splits


def get_topk_from_scores(list1, list2):
    list1 = torch.tensor(list1)
    list2 = torch.tensor(list2)
    _, indices = torch.topk(torch.stack((list1, list2), dim=1), 1)
    return indices


def get_accuracy_score(y_true, id_scores, ood_scores):
    indices = get_topk_from_scores(id_scores, ood_scores)
    return accuracy_score(y_true, indices)


def get_fscore(y_true, id_scores, ood_scores):
    indices = get_topk_from_scores(id_scores, ood_scores)
    return f1_score(y_true=y_true, y_pred=indices, pos_label=1)


def get_mean_std(ls):
    return np.mean(ls), np.std(ls)


@torch.no_grad()
def image_decoder(clip_model,
                  clip_tokenizer,
                  bert_tokenizer,
                  bert_model,
                  device,
                  isolated_classes: IsolatedClasses = None,
                  id_classes=6,
                  ood_classes=4,
                  runs=1, ):
    ablation_splits = get_ablation_splits(isolated_classes.labels, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum, auc_list_mean, auc_list_max = [], [], []
    for split in ablation_splits:
        seen_labels = split[:id_classes]
        unseen_labels = split[id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
        f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

        for i, semantic_label in enumerate(split):
            _logger.info(f"Encoding images for label {semantic_label}")
            loader = isolated_classes[semantic_label]
            for idx, image in enumerate(tqdm(loader)):
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

                text_features = clip_model.encode_text(all_desc_ids.to(device)).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

                # detection score is accumulative sum of probs of generated entities
                ood_prob_sum = np.sum(zeroshot_probs[id_classes:].detach().cpu().numpy())
                ood_probs_sum.append(ood_prob_sum)

                ood_prob_mean = np.mean(zeroshot_probs[id_classes:].detach().cpu().numpy())
                ood_probs_mean.append(ood_prob_mean)

                top_prob, _ = zeroshot_probs.cpu().topk(1, dim=-1)
                ood_probs_max.append(top_prob.detach().numpy())

                id_probs_sum.append(1. - ood_prob_sum)

        targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)
        fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                       targets)
        fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

    metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)

    return metrics


def get_id_datasets(dataset, id_classes, kshots=16):
    _logger.info("Creating train set")
    train_transform = get_train_transform()
    train_dataset = dataset(data_path=Config.DATAPATH,
                            train=True,
                            transform=train_transform)

    id_classes_idxs = [train_dataset.class_to_idx[id_class] for id_class in id_classes]
    imgs, targets = [], []
    for img, targ in zip(train_dataset.data, train_dataset.targets):
        if targ in id_classes_idxs:
            imgs.append(img)
            targets.append(targ)

    train_imgs, val_imgs, train_targets, val_targets = train_test_split(imgs, targets, test_size=.5)
    train_dataset.data = train_imgs
    train_dataset.targets = train_targets
    train_dataset.classes = id_classes

    val_dataset = dataset(data_path=Config.DATAPATH,
                          train=True,
                          transform=train_transform)
    val_dataset.data = val_imgs
    val_dataset.targets = val_targets
    val_dataset.classes = id_classes

    return get_kshot_train_set(train_dataset, kshots), val_dataset


def get_tip_adapter_weights(train_set,
                            val_set,
                            clip_model,
                            train_epoch=20,
                            alpha=1., beta=1.17,
                            lr=0.001, eps=1e-4):
    # only for the id classes

    return get_adapter_weights(train_set, val_set, clip_model, train_epoch=train_epoch, alpha=alpha, beta=beta, lr=lr,
                               eps=eps)


def tip_image_decoder(clip_model,
                      clip_tokenizer,
                      bert_tokenizer,
                      bert_model,
                      device,
                      dataset,
                      isolated_classes: IsolatedClasses,
                      id_classes,
                      ood_classes,
                      runs,
                      kshots=16,
                      train_epoch=2,
                      alpha=1., beta=1.17,
                      lr=0.001, eps=1e-4
                      ):
    ablation_splits = get_ablation_splits(isolated_classes.labels, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum, auc_list_mean, auc_list_max = [], [], []
    for split in ablation_splits:

        seen_labels = split[:id_classes]

        train_set, val_set = get_id_datasets(dataset, seen_labels, kshots=kshots)
        cache_keys, cache_values = get_train_features(train_set, clip_model)

        adapter = WeightAdapter(clip_model, cache_keys).to(device)
        adapter_weights = get_tip_adapter_weights(train_set, val_set,
                                                  clip_model, train_epoch=train_epoch,
                                                  alpha=alpha, beta=beta,
                                                  lr=lr, eps=eps)
        adapter.weights = adapter_weights

        unseen_labels = split[id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
        f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

        for i, semantic_label in enumerate(split):
            _logger.info(f"Encoding images for label {semantic_label}")
            loader = isolated_classes[semantic_label]
            for idx, image in enumerate(tqdm(loader)):
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
                _logger.debug("Semantic label: {semantic_label}Unique Entities: {unique_entities}")

                all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
                all_desc_ids = tokenize_for_clip(all_desc, clip_tokenizer)

                image_feature = clip_out
                image_feature /= image_feature.norm(dim=-1, keepdim=True)

                with torch.no_grad():
                    text_features = clip_model.encode_text(all_desc_ids.to(device)).float()
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    affinity = adapter(image_feature)

                print(f"affinity: {affinity.shape}")
                cache_logits = get_cache_logits(affinity, cache_values, beta)
                print(f"cache_logits: {cache_logits.shape}")

                print(f"image_feature: {image_feature.shape}")
                print(f"text_feature: {text_features.T.shape}")
                clip_logits = 100. * image_feature @ text_features.T  # should work
                print(f"clip_logits: {clip_logits.shape}")

                tip_logits = clip_logits + cache_logits * alpha  # will fal
                zeroshot_probs_clip = clip_logits.softmax(dim=-1).squeeze()
                zeroshot_probs_tip = tip_logits.softmax(dim=-1).squeeze
                raise ValueError
                # detection score is accumulative sum of probs of generated entities
                ood_prob_sum = np.sum(zeroshot_probs_clip[id_classes:].detach().cpu().numpy())
                ood_probs_sum.append(ood_prob_sum)

                ood_prob_mean = np.mean(zeroshot_probs_clip[id_classes:].detach().cpu().numpy())
                ood_probs_mean.append(ood_prob_mean)

                top_prob, _ = zeroshot_probs_clip.cpu().topk(1, dim=-1)
                ood_probs_max.append(top_prob.detach().numpy())

                id_probs_sum.append(1. - ood_prob_sum)

            targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)
            fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                           targets)
            fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

        metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)

        return metrics


def fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets):
    f_score = get_fscore(targets, np.squeeze(id_probs_sum), np.squeeze(ood_probs_sum))
    accuracy = get_accuracy_score(np.array(targets), np.squeeze(id_probs_sum), np.squeeze(ood_probs_sum))
    f_probs_sum.append(f_score)
    acc_probs_sum.append(accuracy)


def fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum, targets):
    auc_list_mean.append(get_auroc_for_ood_probs(targets, ood_probs_mean))
    auc_list_max.append(get_auroc_for_ood_probs(targets, ood_probs_max))
    auc_list_sum.append(get_auroc_for_ood_probs(targets, ood_probs_sum))


def get_auroc_for_ood_probs(targets, means):
    return roc_auc_score(np.array(targets), np.squeeze(means))


def get_split_specific_targets(isolated_classes, seen_labels, unseen_labels):
    len_id_targets = sum([len(isolated_classes[lab].dataset) for lab in seen_labels])
    len_ood_targets = sum([len(isolated_classes[lab].dataset) for lab in unseen_labels])
    targets = torch.tensor(len_id_targets * [0] + len_ood_targets * [1])
    return targets


def get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum):
    sum_mean_f1, sum_std_f1 = get_mean_std(f_probs_sum)
    sum_mean_acc, sum_std_acc = get_mean_std(acc_probs_sum)
    sum_mean_auc, sum_std_auc = get_mean_std(auc_list_sum)
    mean_mean_auc, mean_std_auc = get_mean_std(auc_list_mean)
    max_mean_auc, max_std_auc = get_mean_std(auc_list_max)
    metrics = {'auc-mean': mean_mean_auc,
               'auc-max': max_mean_auc,
               'auc-sum:': sum_mean_auc,
               'f1_mean': sum_mean_f1,
               'acc_mean': sum_mean_acc}
    return metrics

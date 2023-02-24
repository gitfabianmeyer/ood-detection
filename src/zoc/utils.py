import logging
import random
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from clip.simple_tokenizer import SimpleTokenizer
from zeroshot.classification import get_cosine_similarity_matrix_for_normed_features
from adapters.tip_adapter import get_train_transform, get_kshot_set
from ood_detection.config import Config
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.zoc_loader import IsolatedClasses
from transformers import BertGenerationConfig, BertGenerationDecoder
from zeroshot.utils import get_image_features_for_isolated_class_loader, FeatureDict

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


def get_ablation_splits(classnames, n, id_classes, ood_classes):
    if id_classes + ood_classes > len(classnames):
        raise IndexError("Too few classes to build split")

    splits = []
    for _ in range(n):
        base = random.sample(classnames, k=id_classes + ood_classes)
        splits.append(base[:id_classes] + base[-ood_classes:])
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
                  isolated_classes: IsolatedClasses,
                  id_classes,
                  ood_classes,
                  runs):
    ablation_splits = get_ablation_splits(isolated_classes.classes, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum, auc_list_mean, auc_list_max = [], [], []
    for split in ablation_splits:
        seen_labels = split[:id_classes]
        unseen_labels = split[id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

        ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
        for i, semantic_label in enumerate(split):
            loader = isolated_classes[semantic_label]
            for idx, image in enumerate(tqdm(loader)):
                clip_out = clip_model.encode_image(image.to(device)).float()
                ood_prob_max, ood_prob_mean, ood_prob_sum = get_mean_max_sum_for_zoc_image(bert_model, bert_tokenizer,
                                                                                           clip_model, clip_tokenizer,
                                                                                           device, id_classes, clip_out,
                                                                                           seen_descriptions,
                                                                                           seen_labels)

                ood_probs_sum.append(ood_prob_sum)
                ood_probs_mean.append(ood_prob_mean)
                ood_probs_max.append(ood_prob_max.detach().numpy())

        targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)
        fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                       targets)
        fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

    metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)

    return metrics


def get_zoc_probs(image_features, bert_model, bert_tokenizer, clip_model, clip_tokenizer, device, seen_descriptions,
                  seen_labels):
    text_features = get_caption_features_from_image_features(image_features, seen_descriptions,
                                                             seen_labels, bert_model,
                                                             bert_tokenizer, clip_model,
                                                             clip_tokenizer, device)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    zeroshot_probs = get_cosine_similarity_matrix_for_normed_features(image_features, text_features, 100)
    return zeroshot_probs.softmax(dim=1).squeeze()


@torch.no_grad()
def get_caption_features_from_image_features(unnormed_image_feature, seen_descriptions, seen_label, bert_model,
                                             bert_tokenizer,
                                             clip_model, clip_tokenizer, device):
    clip_extended_embed = unnormed_image_feature.repeat(1, 2).type(torch.FloatTensor)
    # greedy generation
    target_list, topk_list = greedysearch_generation_topk(clip_extended_embed,
                                                          bert_tokenizer,
                                                          bert_model,
                                                          device)
    topk_tokens = [bert_tokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]
    unique_entities = list(set(topk_tokens) - {seen_label})
    all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
    all_desc_ids = tokenize_for_clip(all_desc, clip_tokenizer)
    text_features = clip_model.encode_text(all_desc_ids.to(device)).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze()


def get_sum_max_mean_probs(zeroshot_probs, id_classes):
    ood_prob_sum = np.sum(zeroshot_probs[id_classes:].detach().cpu().numpy())
    ood_prob_mean = np.mean(zeroshot_probs[id_classes:].detach().cpu().numpy())
    ood_max_prob, _ = zeroshot_probs.cpu().topk(1, dim=-1)
    return ood_prob_sum, ood_max_prob, ood_prob_mean


@torch.no_grad()
def image_decoder_featuredict(clip_model,
                              clip_tokenizer,
                              bert_tokenizer,
                              bert_model,
                              device,
                              feature_dict,
                              id_classes=6,
                              ood_classes=4,
                              runs=1, ):
    ablation_splits = get_ablation_splits(feature_dict.keys(), n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum, auc_list_mean, auc_list_max = [], [], []
    for split in ablation_splits:
        seen_labels = split[:id_classes]
        unseen_labels = split[id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
        f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

        for semantic_label in split:
            image_features = feature_dict[semantic_label]
            for image in tqdm(image_features):
                ood_prob_max, ood_prob_mean, ood_prob_sum = get_mean_max_sum_for_zoc_image(bert_model, bert_tokenizer,
                                                                                           clip_model, clip_tokenizer,
                                                                                           device, id_classes, image,
                                                                                           seen_descriptions,
                                                                                           seen_labels)

                ood_probs_sum.append(ood_prob_sum)
                ood_probs_mean.append(ood_prob_mean)
                ood_probs_max.append(ood_prob_max.detach().numpy())

        targets = get_split_specific_targets(feature_dict, seen_labels, unseen_labels)
        fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                       targets)
        fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

    metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)

    return metrics


def get_mean_max_sum_for_zoc_image(bert_model, bert_tokenizer,
                                   clip_model, clip_tokenizer,
                                   device, id_classes,
                                   image_features, seen_descriptions,
                                   seen_labels):
    zeroshot_probs = get_zoc_probs(image_features, bert_model,
                                   bert_tokenizer, clip_model,
                                   clip_tokenizer, device,
                                   seen_descriptions, seen_labels)
    # detection score is accumulative sum of probs of generated entities
    ood_prob_sum, ood_prob_max, ood_prob_mean = get_sum_max_mean_probs(zeroshot_probs, id_classes)
    return ood_prob_max, ood_prob_mean, ood_prob_sum


def get_id_datasets(dataset, id_classes, kshots=16):
    # do stuff here
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

    return get_kshot_set(train_dataset, kshots), val_dataset


def fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets):
    f_score = get_fscore(targets, np.squeeze(id_probs_sum), np.squeeze(ood_probs_sum))
    accuracy = get_accuracy_score(np.array(targets), np.squeeze(id_probs_sum), np.squeeze(ood_probs_sum))
    f_probs_sum.append(f_score)
    acc_probs_sum.append(accuracy)


def fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum, targets):
    auc_list_mean.append(get_auroc_for_ood_probs(targets, ood_probs_mean))
    auc_list_max.append(get_auroc_for_max_probs(targets, ood_probs_max))
    auc_list_sum.append(get_auroc_for_ood_probs(targets, ood_probs_sum))


# needed so we can flip the classifier to become > 50%
def get_auroc_for_max_probs(targets, max_probs):
    return roc_auc_score(np.logical_not(np.array(targets)).astype(int), max_probs)


def get_auroc_for_ood_probs(targets, means):
    return roc_auc_score(np.array(targets), np.squeeze(means))


def get_split_specific_targets(isolated_classes, seen_labels, unseen_labels):
    if isinstance(isolated_classes, IsolatedClasses):
        len_id_targets = sum([len(isolated_classes[lab].dataset) for lab in seen_labels])
        len_ood_targets = sum([len(isolated_classes[lab].dataset) for lab in unseen_labels])
        targets = torch.tensor(len_id_targets * [0] + len_ood_targets * [1])
    elif isinstance(isolated_classes, Dict):
        len_id_targets = sum([len(isolated_classes[lab]) for lab in seen_labels])
        len_ood_targets = sum([len(isolated_classes[lab]) for lab in unseen_labels])
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
               'auc-sum': sum_mean_auc,
               'f1_mean': sum_mean_f1,
               'acc_mean': sum_mean_acc}
    return metrics


def get_decoder(clearml_model=False):
    if clearml_model:
        from clearml import Task
        artifact_task = Task.get_task(project_name='ma_fmeyer', task_name='Train Decoder')

        model_path = artifact_task.artifacts['model'].get_local_copy()
    else:
        model_path = Config.MODEL_PATH

    bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    bert_config.is_decoder = True
    bert_config.add_cross_attention = True
    bert_model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                       config=bert_config).to(Config.DEVICE).eval()

    bert_model.load_state_dict(
        torch.load(model_path + 'model_3.pt', map_location=torch.device(Config.DEVICE))['net'])

    return bert_model


def get_feature_dict_from_isolated_classes(isolated_classes: IsolatedClasses, clip_model, max_len=50000):
    _logger.info("Start creating image features...")
    max_per_class = max_len // len(isolated_classes.classes)
    feature_dict = {}
    for cls in tqdm(isolated_classes.classes):
        feature_dict[cls] = get_image_features_for_isolated_class_loader(isolated_classes[cls], clip_model,
                                                                         max_per_class)

    return feature_dict


@torch.no_grad()
def get_zoc_unique_entities(dataset, clip_model, bert_tokenizer, bert_model, device):
    isolated_classes_slow_loader = IsolatedClasses(dataset,
                                                   batch_size=1,
                                                   lsun=False)
    zoc_unique_entities = defaultdict(list)

    for semantic_label in tqdm(dataset.classes):
        loader = isolated_classes_slow_loader[semantic_label]
        for image_idx, image in enumerate(loader):
            clip_out = clip_model.encode_image(image.to(device)).float()
            clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)

            # greedy generation
            target_list, topk_list = greedysearch_generation_topk(clip_extended_embed,
                                                                  bert_tokenizer,
                                                                  bert_model,
                                                                  device)

            topk_tokens = [bert_tokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]
            unique_entities = list(set(topk_tokens) - {semantic_label})
            zoc_unique_entities[semantic_label].append(unique_entities)

    return zoc_unique_entities


def get_zoc_feature_dict(dataset, clip_model):
    device = Config.DEVICE
    isolated_classes = IsolatedClasses(dataset,
                                       batch_size=512,
                                       lsun=False)
    image_featuredict = get_unnormed_featuredict_from_isolated_classes(clip_model, device, isolated_classes)

    from transformers import BertGenerationTokenizer
    bert_tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    clip_tokenizer = SimpleTokenizer()
    _logger.info('Loading decoder model')
    bert_model = get_decoder()

    seen_labels = image_featuredict.classes
    seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

    zoc_featuredict = {}
    for semantic_label, image_features in image_featuredict.items():
        text_features = []
        for image_feat in image_features:
            tf = get_caption_features_from_image_features(image_feat, seen_descriptions,
                                                          semantic_label, bert_model,
                                                          bert_tokenizer, clip_model,
                                                          clip_tokenizer, device)

            text_features.append(tf)
        zoc_featuredict[semantic_label] = torch.cat(text_features, dim=0)

    return FeatureDict(zoc_featuredict, None)


@torch.no_grad()
def get_unnormed_featuredict_from_isolated_classes(clip_model, device, isolated_classes):
    feature_dict = {}
    for cls in tqdm(isolated_classes.classes):
        loader = isolated_classes[cls]

        label_feats = []
        for images in loader:
            images = images.to(device)
            image_features = clip_model.encode_image(images)
            label_feats.append(image_features)
        feature_dict[cls] = torch.cat(label_feats).half()

    return FeatureDict(feature_dict, None)
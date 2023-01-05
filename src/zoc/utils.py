import logging
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from datasets.zoc_loader import IsolatedClasses

_logger = logging.getLogger()


@torch.no_grad()
def greedysearch_generation_topk(clip_embed, berttokenizer, bert_model, device):
    N = 1  # batch has single sample
    max_len = 77
    target_list = [torch.tensor(berttokenizer.bos_token_id)]
    top_k_list = []
    bert_model.eval()
    for i in range(max_len):
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
    for i in range(n):
        base = random.choices(classnames, k=id_classes)
        leftover = [classname for classname in classnames if classname not in base]
        oods = random.choices(leftover, k=ood_classes)
        splits.append(base + oods)

    return splits


def get_accuracy_score(y_true, id_scores, ood_scores):
    _, indices = torch.topk(torch.stack(id_scores, ood_scores), 1)
    return accuracy_score(y_true, indices)


@torch.no_grad()
def image_decoder(clip_model,
                  clip_tokenizer,
                  bert_tokenizer,
                  bert_model,
                  device,
                  isolated_classes: IsolatedClasses = None,
                  id_classes=6,
                  ood_classes=4,
                  runs=1):
    ablation_splits = get_ablation_splits(isolated_classes.labels, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum = []
    for split in ablation_splits:
        seen_labels = split[:id_classes]
        unseen_labels = split[id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        ood_probs_sum, f_probs_sum, acc_probs_sum, id_probs_sum = [], [], [], []

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
                _logger.debug("Semantic label: {semantic_label}Unique Entities: {unique_entities}")

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

                id_probs_sum.append(1. - ood_prob_sum)

        len_id_targets = sum([len(isolated_classes[lab].dataset) for lab in seen_labels])
        len_ood_targets = sum([len(isolated_classes[lab].dataset) for lab in unseen_labels])
        targets = torch.tensor(len_id_targets * [0] + len_ood_targets * [1])

        auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))
        f_score = f1_score(np.array(targets), np.squeeze(ood_probs_sum))
        accuracy = get_accuracy_score(np.array(targets), np.squeeze(id_probs_sum), np.squeeze(ood_probs_sum))

        auc_list_sum.append(auc_sum)
        f_probs_sum.append(f_score)
        acc_probs_sum.append(accuracy)

    std_auc, mean_auc = torch.std_mean(auc_list_sum)
    std_f1, mean_f1 = torch.std_mean(f_probs_sum)
    std_acc, mean_acc = torch.std_mean(acc_probs_sum)

    metrics = {'auc_mean': mean_auc,
               'auc_std': std_auc,
               'f1_std': std_f1,
               'f1_mean': mean_f1,
               'acc_std': std_acc,
               'acc_mean': mean_acc}

    return metrics

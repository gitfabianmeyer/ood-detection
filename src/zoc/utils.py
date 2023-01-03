import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from datasets.zoc_loader import IsolatedClasses


def greedysearch_generation_topk(clip_embed, berttokenizer, bert_model, device):
    N = 1  # batch has single sample
    max_len = 77
    target_list = [torch.tensor(berttokenizer.bos_token_id)]
    top_k_list = []
    bert_model.eval()
    for i in range(max_len):
        target = torch.LongTensor(target_list).unsqueeze(0)
        position_ids = torch.arange(0, len(target)).expand(N, len(target)).to(device)
        with torch.no_grad():
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
    for i in range(n):
        base = random.choices(classnames, k=id_classes)
        leftover = [classname for classname in classnames if classname not in base]
        oods = random.choices(leftover, k=ood_classes)
        splits.append(base + oods)

    return splits


@torch.no_grad()
def image_decoder(clip_model,
                  clip_tokenizer,
                  bert_tokenizer,
                  bert_model,
                  device,
                  classnames,
                  isolated_classes: IsolatedClasses = None,
                  id_classes=6,
                  ood_classes=4):
    ablation_splits = get_ablation_splits(classnames, n=10, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum = []
    for split in ablation_splits:
        seen_labels = split[:id_classes]
        unseen_labels = split[id_classes:]
        print(f"Seen labels: {seen_labels}")
        print(f"OOD Labels: {split[id_classes:]}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        len_id_targets = sum([len(isolated_classes[lab].dataset) for lab in seen_labels])
        len_ood_targets = sum([len(isolated_classes[lab].dataset) for lab in unseen_labels])

        max_num_entities = 0
        ood_probs_sum = []
        for i, semantic_label in enumerate(split):
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
                if len(unique_entities) > max_num_entities:
                    max_num_entities = len(unique_entities)
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

        targets = torch.tensor(len_id_targets * [0] + len_ood_targets * [1])

        auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))
        print('sum_ood AUROC={}'.format(auc_sum))
        auc_list_sum.append(auc_sum)
    print('all auc scores:', auc_list_sum)
    mean_auc = np.mean(auc_list_sum)
    std_auc = np.std(auc_list_sum)
    print('auc sum', mean_auc, std_auc)
    return mean_auc, std_auc

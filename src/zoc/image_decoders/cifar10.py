import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from zoc.utils import tokenize_for_clip, greedysearch_generation_topk


def image_decoder(clip_model, clip_tokenizer, berttokenizer, device, image_loaders=None):
    splits = [['airplane', 'automobile', 'truck', 'horse', 'cat', 'bird', 'ship', 'deer', 'dog', 'frog'],
              ['airplane', 'automobile', 'truck', 'horse', 'cat', 'bird', 'ship', 'deer', 'dog', 'frog'],
              ['airplane', 'bird', 'deer', 'cat', 'horse', 'dog', 'ship', 'automobile', 'frog', 'truck'],
              ['dog', 'automobile', 'truck', 'ship', 'horse', 'airplane', 'bird', 'cat', 'deer', 'frog'],
              ['dog', 'horse', 'automobile', 'ship', 'deer', 'frog', 'airplane', 'truck', 'bird', 'cat'],
              ['ship', 'automobile', 'dog', 'cat', 'deer', 'frog', 'airplane', 'truck', 'bird', 'horse']]
    ablation_splits = [['airplane', 'automobile', 'truck', 'horse', 'cat', 'bird', 'ship', 'dog', 'deer', 'frog'],
                       ['airplane', 'automobile', 'truck', 'bird', 'ship', 'frog', 'deer', 'dog', 'horse', 'cat']]
    # ablation_splits = [['horse', 'cat', 'deer', 'frog'],
    #                   ['deer', 'frog', 'horse', 'cat']]

    auc_list_sum = []
    for split in ablation_splits:
        seen_labels = split[:8]
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]
        # targets = torch.tensor(6000*[0] + 4000*[1])
        targets = torch.tensor(8000 * [0] + 2000 * [1])
        # targets = torch.tensor(20 * [0] + 20 * [1])
        max_num_entities = 0
        ood_probs_sum = []
        for i, semantic_label in enumerate(split):
            loader = image_loaders[semantic_label]
            for idx, image in enumerate(tqdm(loader)):
                # if idx==10:break
                with torch.no_grad():
                    clip_out = clip_model.encode_image(image.to(device)).float()
                    clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)

                    # greedy generation
                    target_list, topk_list = greedysearch_generation_topk(clip_extended_embed)

                    target_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in target_list]
                    topk_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]

                    unique_entities = list(set(topk_tokens) - {semantic_label})
                    if len(unique_entities) > max_num_entities:
                        max_num_entities = len(unique_entities)
                    all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
                    all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

                    image_feature = clip_model.encode_image(image.cuda()).float()
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)
                    text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

                # detection score is accumulative sum of probs of generated entities
                ood_prob_sum = np.sum(zeroshot_probs[6:].detach().cpu().numpy())
                ood_probs_sum.append(ood_prob_sum)
        auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))
        print('sum_ood AUROC={}'.format(auc_sum))
        auc_list_sum.append(auc_sum)
    print('all auc scores:', auc_list_sum)
    print('auc sum', np.mean(auc_list_sum), np.std(auc_list_sum))
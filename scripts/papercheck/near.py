import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from datasets.zoc_loader import IsolatedClasses
from ood_detection.ood_utils import sorted_zeroshot_weights
from zoc.baseline import get_feature_weight_dict, get_zeroshot_weight_dict


import logging

import clip
import numpy as np
import torch
import wandb
from clip.simple_tokenizer import SimpleTokenizer
from datasets.config import DATASETS_DICT
from ood_detection.classification_utils import zeroshot_classifier
from ood_detection.config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertGenerationTokenizer
from zoc.utils import greedysearch_generation_topk, tokenize_for_clip, get_auroc_for_ood_probs, get_auroc_for_max_probs, \
    get_decoder, get_ablation_splits, get_split_specific_targets

_logger = logging.getLogger(__name__)


@torch.no_grad()
def clip_near_ood_temperatures(clip_model,
                               device,
                               isolated_classes: IsolatedClasses,
                               id_split,
                               runs,
                               min_temp,
                               max_temp,
                               num_temps
                               ):
    len_all_classes = len(isolated_classes.classes)
    id_classes = int(len_all_classes * id_split)
    ood_classes = len_all_classes - id_classes
    feature_weight_dict = get_feature_weight_dict(isolated_classes, clip_model, device)
    classes_weight_dict = get_zeroshot_weight_dict(isolated_classes, clip_model)

    ablation_splits = get_ablation_splits(isolated_classes.classes, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    # for each temperature..
    for temperature in np.logspace(np.log2(min_temp), np.log2(max_temp), num=num_temps,
                                   base=2.0):  # 10 values between .007 and 100

        split_aurocs = []
        for split in ablation_splits:

            seen_labels = split[:id_classes]
            unseen_labels = split[id_classes:]
            _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")

            zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)

            ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
            f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

            # do 10 times
            for i, semantic_label in enumerate(split):
                # get features
                image_features_for_label = feature_weight_dict[semantic_label]
                # calc the logits and softmaxs
                zeroshot_probs = (temperature * image_features_for_label.to(torch.float32) @ zeroshot_weights.T.to(
                    torch.float32)).softmax(dim=-1).squeeze()

                assert zeroshot_probs.shape[1] == id_classes
                # detection score is accumulative sum of probs of generated entities
                # careful, only for this setting axis=1
                ood_prob_sum = np.sum(zeroshot_probs.detach().cpu().numpy(), axis=1)
                ood_probs_sum.extend(ood_prob_sum)

                ood_prob_mean = np.mean(zeroshot_probs.detach().cpu().numpy(), axis=1)
                ood_probs_mean.extend(ood_prob_mean)

                top_prob, _ = zeroshot_probs.cpu().topk(1, dim=-1)
                ood_probs_max.extend(top_prob.detach().numpy())

                id_probs_sum.extend(1. - ood_prob_sum)

            targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)
            split_auroc = get_auroc_for_max_probs(targets, ood_probs_max)
            split_aurocs.append(split_auroc)

        result = {'clip': np.mean(split_aurocs),
                  'std': np.std(split_aurocs),
                  'temperature': temperature}
        wandb.log(result)
    return True


def main():
    datasets = DATASETS_DICT
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    device = Config.DEVICE

    for dname, dset in tqdm(datasets.items()):
        isolated_classes = IsolatedClasses(dset(Config.DATAPATH,
                                                transform=clip_transform,
                                                split='test'),
                                           batch_size=512)
        run = wandb.init(project=f"thesis-near_ood-temp",
                         entity="wandbefab",
                         name=dname)
        results = clip_near_ood_temperatures(clip_model,
                                             device,
                                             isolated_classes,
                                             Config.ID_SPLIT,
                                             50,
                                             0.01,
                                             100.,
                                             50, )
        run.finish()


if __name__ == '__main__':
    main()

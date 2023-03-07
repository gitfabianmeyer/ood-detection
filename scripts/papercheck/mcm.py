import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import logging
from zeroshot.classification import get_cosine_similarity_matrix_for_normed_features
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
from zoc.utils import get_auroc_for_ood_probs, get_auroc_for_max_probs, \
    get_decoder,get_caption_features_from_image_features

_logger = logging.getLogger(__name__)
datasets = DATASETS_DICT


@torch.no_grad()
def get_zoc_scores(in_distribution, loader, seen_labels, clip_model, clip_tokenizer, bert_tokenizer, bert_model,
                   device):
    id_length = len(seen_labels)
    seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]
    ood_probs_sum = []
    for idx, (image, target) in enumerate(tqdm(loader)):
        clip_out = clip_model.encode_image(image.to(device)).float()
        text_features = get_caption_features_from_image_features(clip_out, seen_descriptions, seen_labels, bert_model,
                                                                 bert_tokenizer, clip_model, clip_tokenizer)
        image_feature = clip_out
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        zeroshot_probs = (100.0 * image_feature.to(torch.float32) @ text_features.T.to(torch.float32)).softmax(
            dim=-1).squeeze()

        # detection score is accumulative sum of probs of generated entities
        ood_prob_sum = np.sum(zeroshot_probs[id_length:].detach().cpu().numpy())
        ood_probs_sum.append(ood_prob_sum)

    return ood_probs_sum


@torch.no_grad()
def get_clip_zeroshot_score(clip_model, id_set, ood_set, temperature):
    device = Config.DEVICE
    zeroshot_weights = zeroshot_classifier(id_set.classes, id_set.templates, clip_model)

    top_probs = []
    for dataset in [id_set, ood_set]:
        image_features_full = []
        loader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=3)

        for imgs, _ in loader:
            imgs = imgs.to(device)

            image_features = clip_model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features_full.append(image_features)

        image_features_full = torch.cat(image_features_full, dim=0)
        zsw = get_cosine_similarity_matrix_for_normed_features(image_features_full, zeroshot_weights, temperature)
        clip_probs = torch.softmax(zsw, dim=-1).squeeze()
        top_clip_prob, _ = clip_probs.cpu().topk(1, dim=-1)
        top_probs.extend(top_clip_prob)

    top_probs = torch.stack(top_probs).squeeze()
    targets = torch.Tensor([0] * len(id_set) + [1] * len(ood_set))
    score = get_auroc_for_max_probs(targets, top_probs)
    _logger.info(f"Clip AUROC: {score:.3f}")
    return score


@torch.no_grad()
def run_zoc_and_clip(id_set, ood_set, temp):
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    id_dataset = id_set(Config.DATAPATH,
                        transform=clip_transform,
                        split='test')
    ood_dataset = ood_set(Config.DATAPATH,
                          transform=clip_transform,
                          split='test')

    # _logger.info("Running zoc")
    # auroc_zoc_score = get_all_zoc_scores(clip_model, id_dataset, ood_dataset)
    _logger.info("Running clip")
    auroc_clip_score = get_clip_zeroshot_score(clip_model, id_dataset, ood_dataset, temp)

    results = {'clip': np.mean(auroc_clip_score), 'clip_std': np.std(auroc_clip_score), 'temp': temp}
               # 'zoc': np.mean(auroc_zoc_score),
               # 'zoc_std': np.std(auroc_zoc_score)}

    return results


def get_all_zoc_scores(clip_model, id_dataset, ood_set):
    bert_tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    clip_tokenizer = SimpleTokenizer()
    bert_model = get_decoder()
    device = Config.DEVICE
    id_loader = DataLoader(id_dataset,
                           batch_size=1,
                           shuffle=False)
    ood_loader = DataLoader(ood_set,
                            batch_size=1,
                            shuffle=False)
    seen_labels = id_dataset.classes
    _logger.info(f"Encoding images for ID")
    id_scores = get_zoc_scores(True, id_loader, seen_labels, clip_model, clip_tokenizer, bert_tokenizer, bert_model,
                               device)
    _logger.info(f"Encoding images for OOD")
    ood_scores = get_zoc_scores(False, ood_loader, seen_labels, clip_model, clip_tokenizer, bert_tokenizer, bert_model,
                                device)

    full = id_scores + ood_scores
    targets = torch.Tensor([0] * len(id_scores) + [1] * len(ood_scores))
    auroc_zoc_score = get_auroc_for_ood_probs(targets, full)
    _logger.info(f"ZOC AUROC: {auroc_zoc_score:.3f}")

    return auroc_zoc_score


def main():
    in_dist = 'caltech cub'
    oo_dist = 'dtd'
    id_set = datasets[in_dist]
    ood_set = datasets[oo_dist]

    run = wandb.init(project=f"thesis-far_ood-temp",
                     entity="wandbefab",
                     name=f"{in_dist}_{oo_dist}")

    for temp in np.linspace(1., 100., 50):
        results = run_zoc_and_clip(id_set, ood_set, temp)
        wandb.log(results)
    run.finish()


if __name__ == '__main__':
    main()

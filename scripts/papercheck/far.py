import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import logging

import clip
import numpy as np
import torch
import wandb
from clip.simple_tokenizer import SimpleTokenizer
from datasets.config import DATASETS_DICT, HalfOneDict
from ood_detection.classification_utils import zeroshot_classifier
from ood_detection.config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertGenerationTokenizer
from zoc.utils import greedysearch_generation_topk, tokenize_for_clip, get_auroc_for_ood_probs, get_auroc_for_max_probs, \
    get_decoder

_logger = logging.getLogger(__name__)
datasets = DATASETS_DICT


@torch.no_grad()
def get_clip_auroc_from_features(id_features, ood_features, zeroshot_weights, temperature):
    top_probs = []
    for features in [id_features, ood_features]:
        zsw = temperature * features.to(torch.float32) @ zeroshot_weights.T
        clip_probs = torch.softmax(zsw, dim=-1).squeeze()
        top_clip_prob, _ = clip_probs.cpu().topk(1, dim=-1)
        top_probs.extend(top_clip_prob)

    top_probs = torch.stack(top_probs).squeeze()
    targets = torch.Tensor([0] * len(id_features) + [1] * len(ood_features))
    score = get_auroc_for_max_probs(targets, top_probs)
    return score

@torch.no_grad()
def get_set_features(dataset, clip_model):
    device = Config.DEVICE
    loader = DataLoader(dataset,
                        batch_size=512)
    image_features_full = []
    for imgs, _ in loader:
        imgs = imgs.to(device)

        image_features = clip_model.encode_image(imgs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features_full.append(image_features)

    return torch.cat(image_features_full, dim=0)


def main():
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    temperatures = np.logspace(1, 100, num=25, base=2.0)
    for id_name, id_set in HalfOneDict.items():
        id_dataset = id_set(Config.DATAPATH,
                            transform=clip_transform,
                            split='test')

        id_set_features = get_set_features(id_dataset, clip_model)
        zeroshot_weights = zeroshot_classifier(id_set.classes, ['a photo of a {}'], clip_model)

        for od_name, od_set in DATASETS_DICT.items():
            if od_name == id_name:
                continue
            run = wandb.init(project=f"thesis-far-{id_name}",
                             entity="wandbefab",
                             name=od_name)
            ood_dataset = od_set(Config.DATAPATH,
                                 transform=clip_transform,
                                 split='test')

            ood_set_features = get_set_features(ood_dataset, clip_model)

            for temp in temperatures:
                auroc_clip_score = get_clip_auroc_from_features(id_set_features, ood_set_features, zeroshot_weights)
                wandb.log({'clip': auroc_clip_score,
                           'temperature': temp})
            run.finish()


if __name__ == '__main__':
    main()

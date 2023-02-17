import os

from adapters.oodd import get_cosine_similarity_matrix_for_normed_features

from src.ood_detection.baseline import get_trained_linear_classifier
from src.zoc.ablation import linear_adapter_zoc_ablation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import logging
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import wandb

from adapters.tip_adapter import create_tip_train_set, get_cache_model, get_dataset_with_shorted_classes, \
    get_dataset_features_from_dataset_with_split, run_tip_adapter_finetuned, WeightAdapter, load_adapter, search_hp, \
    get_cache_logits
from datasets.zoc_loader import IsolatedClasses
from ood_detection.ood_utils import sorted_zeroshot_weights
from zoc.baseline import get_feature_weight_dict, get_zeroshot_weight_dict, train_id_classifier, FeatureSet

from adapters.ood import get_ablation_split_classes, pad_list_of_vectors

from clip.simple_tokenizer import SimpleTokenizer
from transformers import BertGenerationTokenizer
from zoc.utils import get_decoder, get_ablation_splits, get_zoc_unique_entities, tokenize_for_clip, \
    get_auroc_for_ood_probs, get_auroc_for_max_probs, get_mean_std, get_split_specific_targets
import clip
from ood_detection.config import Config
from datasets.config import KshotSets

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

run_clearml = False
runs = 10
kshots = [2, 4, 6, 8, 16, 32, 64]
train_epochs = 20
augment_epochs = 10
lr = 0.001
eps = 1e-4


def main():
    failed = []
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    device = Config.DEVICE

    for dname, dset in KshotSets.items():
        if dname in ['svhn', 'mnist', 'fashion mnist']:
            continue

        _logger.info(f"\t\tStarting {dname} run...")
        run = wandb.init(project=f"thesis-kshot-linear_ablation-{runs}-runs",
                         entity="wandbefab",
                         name=dname)
        try:
            results = linear_adapter_zoc_ablation(dset,
                                                  clip_model,
                                                  clip_transform,
                                                  device,
                                                  Config.ID_SPLIT,
                                                  augment_epochs,
                                                  runs,
                                                  kshots,
                                                  train_epochs,
                                                  lr,
                                                  eps)
            wandb.log(results)
        except Exception as e:
            failed.append(dname)
            raise e
        run.finish()
    print(f"Failed: {failed}")


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

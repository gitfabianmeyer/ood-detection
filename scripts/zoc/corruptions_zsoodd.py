import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datasets import corruptions
from datasets.cifar import OodCifar10
from datasets.gtsrb import OodGTSRB
from datasets.corruptions import get_corruption_transform

import argparse
import logging

import clip
from clip.simple_tokenizer import SimpleTokenizer

import torch
import wandb
from clearml import Task
from datasets.config import DATASETS_DICT, HalfOneDict
from datasets.zoc_loader import IsolatedClasses
from metrics.metrics_logging import wandb_log
from ood_detection.config import Config
from transformers import BertGenerationTokenizer, BertGenerationConfig, BertGenerationDecoder
from zoc.utils import image_decoder

_logger = logging.getLogger(__name__)
splits = [(.4, .6), ]
clearml_model = False
MODEL_PATH = "/home/fmeyer/ZOC/trained_models/COCO/ViT-B32/"


def run_single_dataset_ood(isolated_classes, clip_model, clip_tokenizer, bert_tokenizer, bert_model,
                           id_classes=.6, runs=5):
    labels = isolated_classes.classes
    id_classes = int(len(labels) * id_classes)
    ood_classes = len(labels) - id_classes
    metrics = image_decoder(clip_model=clip_model,
                            clip_tokenizer=clip_tokenizer,
                            bert_tokenizer=bert_tokenizer,
                            bert_model=bert_model,
                            device=Config.DEVICE,
                            isolated_classes=isolated_classes,
                            id_classes=id_classes,
                            ood_classes=ood_classes,
                            runs=runs)
    metrics['num_runs'] = runs

    return metrics


def run_all(args):
    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    bert_tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    clip_tokenizer = SimpleTokenizer()
    bert_model = get_decoder()

    for dname, dset in HalfOneDict.items():

        _logger.info(f"Running {dname} for {args.runs_ood} runs...")

        for corr_name, corr in corruptions.Corruptions.items():
            for severity in [1, 3, 5]:
                _logger.info(f"Running {corr_name} - {severity} ...")
                if dname == 'lsun':
                    lsun = True

                else:
                    lsun = False

                transform = get_corruption_transform(clip_transform, corr, severity)
                isolated_classes = IsolatedClasses(dataset=dset(data_path=Config.DATAPATH,
                                                                train=False,
                                                                transform=transform),
                                                   lsun=lsun)

                for split in splits:
                    # perform zsoodd
                    metrics_dict = run_single_dataset_ood(isolated_classes=isolated_classes,
                                                          clip_model=clip_model,
                                                          clip_tokenizer=clip_tokenizer,
                                                          bert_tokenizer=bert_tokenizer,
                                                          bert_model=bert_model,
                                                          id_classes=split[0],
                                                          runs=args.runs_ood)
                    metrics_dict['corruption'] = corr_name
                    metrics_dict['severity'] = severity
                    metrics_dict['dataset'] = dname
                    metrics_dict['model'] = Config.VISION_MODEL
                    metrics_dict['id split'] = split[0]

                    run = wandb_log(metrics_dict=metrics_dict,
                                    experiment='zsoodd_corruptions')

            run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_ood', type=int, default=1)

    args = parser.parse_args()
    run_all(args)

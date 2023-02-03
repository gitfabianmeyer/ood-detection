import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
runs = 10

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

        # if dname == 'caltech101':
        #     print(f"Jumping over {dname}")
        #     continue

        _logger.info(f"Running {dname}...")

        if dname == 'lsun':
            lsun = True

        else:
            lsun = False

        isolated_classes = IsolatedClasses(dataset=dset(data_path=Config.DATAPATH,
                                                        split='test',
                                                        transform=clip_transform),
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
            metrics_dict['id split'] = split[0]
            run = wandb.init(project="thesis-zsoodd_all_classes_five_runs",
                             entity="wandbefab",
                             name=dname + ' split-' + str(split))
            wandb.log(metrics_dict)
            run.finish()
            # print(metrics_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_ood', type=int, default=5)

    args = parser.parse_args()
    run_all(args)

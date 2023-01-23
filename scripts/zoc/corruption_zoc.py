import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets.corruptions import get_corruption_transform, THESIS_CORRUPTIONS, store_corruptions_feature_dict, \
    load_corruptions_feature_dict

import argparse
import logging

import clip
from clip.simple_tokenizer import SimpleTokenizer

import torch
import wandb
from clearml import Task
from datasets.config import CorruptionSets
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config
from transformers import BertGenerationTokenizer, BertGenerationConfig, BertGenerationDecoder
from zoc.utils import image_decoder, get_decoder, image_decoder_featuredict, get_feature_dict_from_isolated_classes

_logger = logging.getLogger(__name__)
splits = [(.4, .6), ]
clearml_model = False
create_features = True


def run_single_dataset_ood(feature_dict, clip_model, clip_tokenizer, bert_tokenizer, bert_model,
                           id_classes=.6, runs=5):
    labels = list(feature_dict.keys())
    _logger.info(f'Running with classes {labels[:10]} ...')
    id_classes = int(len(labels) * id_classes)
    ood_classes = len(labels) - id_classes
    metrics = image_decoder_featuredict(clip_model=clip_model,
                                        clip_tokenizer=clip_tokenizer,
                                        bert_tokenizer=bert_tokenizer,
                                        bert_model=bert_model,
                                        device=Config.DEVICE,
                                        feature_dict=feature_dict,
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

    _logger.info('Loading decoder model')
    bert_model = get_decoder(clearml_model, Config.MODEL_PATH)

    for dname, dset in CorruptionSets.items():
        for cname, ccorr in THESIS_CORRUPTIONS.items():

            run = wandb.init(project="thesis-zoc-selected_corruption-selected_sets_all-classes",
                             entity="wandbefab",
                             name="_".join([dname, cname]),
                             tags=['distance',
                                   'metrics'])
            for severity in [1, 3, 5]:
                _logger.info(f"Running {dname} with {cname} and severity {severity}")

                if dname == 'lsun':
                    lsun = True

                else:
                    lsun = False

                transform = get_corruption_transform(clip_transform, ccorr, severity)

                isolated_classes = IsolatedClasses(dataset=dset(data_path=Config.DATAPATH,
                                                                split='test',
                                                                transform=transform),
                                                   lsun=lsun)

                if create_features:
                    _logger.info('Creating corruptions set')
                    feature_dict = get_feature_dict_from_isolated_classes(isolated_classes)
                    store_corruptions_feature_dict(feature_dict, cname, dname + '-test', severity)

                else:
                    _logger.info("Loading feature dict...")
                    feature_dict = load_corruptions_feature_dict(isolated_classes.classes, cname, dname + '-test',
                                                                 severity)
                for split in splits:
                    # perform zsoodd
                    metrics_dict = run_single_dataset_ood(feature_dict=feature_dict,
                                                          clip_model=clip_model,
                                                          clip_tokenizer=clip_tokenizer,
                                                          bert_tokenizer=bert_tokenizer,
                                                          bert_model=bert_model,
                                                          id_classes=split[0],
                                                          runs=args.runs_ood)
                    metrics_dict['severity'] = severity
                    metrics_dict['id split'] = split[0]

                    wandb.log(metrics_dict)

            run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_ood', type=int, default=1)

    args = parser.parse_args()
    run_all(args)

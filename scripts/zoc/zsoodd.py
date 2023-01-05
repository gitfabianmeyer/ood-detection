import argparse

import clip
from clip.simple_tokenizer import SimpleTokenizer

import torch
import wandb
from datasets.config import DATASETS_DICT
from datasets.zoc_loader import IsolatedClasses
from metrics.metrics_logging import wandb_log
from ood_detection.config import Config
from transformers import BertGenerationTokenizer, BertGenerationConfig, BertGenerationDecoder
from zoc.utils import image_decoder

splits = [(.4, .6), ]


def run_single_dataset_ood(isolated_classes, clip_model, clip_tokenizer, bert_tokenizer, bert_model,
                           id_classes=.4, runs=1):
    id_classes = int(len(isolated_classes.labels) * id_classes)
    ood_classes = len(isolated_classes.labels) - id_classes
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
    bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    bert_config.is_decoder = True
    bert_config.add_cross_attention = True
    bert_model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                       config=bert_config).to(Config.DEVICE).train()
    bert_model.load_state_dict(
        torch.load(args.trained_path + args.model_name, map_location=torch.device(Config.DEVICE))['net'])
    for dname, dset in DATASETS_DICT.items():

        if dname != 'cifar10':
            print(f"jumping over {dname}")
            continue

        if dname == 'lsun':
            lsun = True

        else:
            lsun = False

        isolated_classes = IsolatedClasses(dataset=dset(data_path=Config.DATAPATH,
                                                        train=False,
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
                                                  runs=1)
            metrics_dict['dataset'] = dname
            metrics_dict['model'] = Config.VISION_MODEL
            metrics_dict['id split'] = split[0]

            run = wandb_log(metrics_dict=metrics_dict,
                            experiment='zsoodd')
        run.finish()
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_path', type=str, default='/mnt/c/Users/fmeyer/Git/ood-detection/data/zoc/trained_models/COCO/')
    parser.add_argument('--model_name', type=str, default='model_3.pt')
    args = parser.parse_args()
    run_all(args)

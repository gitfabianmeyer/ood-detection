import logging

_logger = logging.getLogger(__name__)


def run_single_dataset_ood(isolated_classes, clip_model, clip_tokenizer, bert_tokenizer, bert_model,
                           id_classes=.6, runs=5):

    from zoc.utils import image_decoder
    from ood_detection.config import Config

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
    return metrics


def run_all(args):
    import clip
    from clip.simple_tokenizer import SimpleTokenizer
    import random
    import numpy as np
    import wandb
    from datasets.config import DATASETS_DICT
    from datasets.zoc_loader import IsolatedClasses
    from ood_detection.config import Config
    from transformers import BertGenerationTokenizer
    from zoc.utils import get_decoder

    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    bert_tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    clip_tokenizer = SimpleTokenizer()
    bert_model = get_decoder()

    assert args.split in range(5), f'Way too high split {args.split}'

    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), 4)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")
    for dname in datasets:
        dset = DATASETS_DICT[dname]
        _logger.info(f"Running {dname}...")
        dataset = dset(data_path=Config.DATAPATH,
                       split='test',
                       transform=clip_transform)

        if args.shorten:
            _logger.warning("USE SHORTENED CLASSSES")
            shorted_classes = random.sample(dataset.classes, 10)
            dataset.classes = shorted_classes
        isolated_classes = IsolatedClasses(dataset)
        run = wandb.init(project=f"thesis-zsoodd_{args.runs}_runs",
                         entity="wandbefab",
                         name=dname)
        # perform zsoodd
        metrics_dict = run_single_dataset_ood(isolated_classes=isolated_classes,
                                              clip_model=clip_model,
                                              clip_tokenizer=clip_tokenizer,
                                              bert_tokenizer=bert_tokenizer,
                                              bert_model=bert_model,
                                              id_classes=Config.ID_SPLIT,
                                              runs=args.runs)

        wandb.log(metrics_dict)
        run.finish()
        # print(metrics_dict)


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--split', type=int)
    parser.add_argument("--shorten", type=bool, default=False)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_all(args)


if __name__ == '__main__':
    main()

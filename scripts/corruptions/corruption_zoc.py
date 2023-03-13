import logging

_logger = logging.getLogger(__name__)


def run_single_dataset_ood(isolated_classes, clip_model, clip_tokenizer, bert_tokenizer, bert_model, runs, shorten):
    from ood_detection.config import Config
    from zoc.detectors import zoc_detector

    metrics = zoc_detector(isolated_classes,
                           clip_model,
                           clip_tokenizer,
                           bert_tokenizer,
                           bert_model,
                           Config.ID_SPLIT,
                           runs,
                           shorten_classes=None if shorten == 0 else shorten)
    return metrics


def run_all(args):
    import numpy as np

    from datasets.corruptions import get_corruption_transform, THESIS_CORRUPTIONS
    import clip
    from clip.simple_tokenizer import SimpleTokenizer
    import wandb
    from datasets.config import CorruptionSets
    from datasets.zoc_loader import IsolatedClasses
    from ood_detection.config import Config
    from transformers import BertGenerationTokenizer
    from zoc.utils import get_decoder

    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    bert_tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    clip_tokenizer = SimpleTokenizer()
    bert_model = get_decoder()

    if args.split == 0:
        datasets = CorruptionSets.keys()
    else:
        datasets_splits = np.array_split(list(CorruptionSets.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")

    for dname in datasets:
        dset = CorruptionSets[dname]
        for cname, ccorr in THESIS_CORRUPTIONS.items():

            run = wandb.init(project="thesis-corruptions-zoc",
                             entity="wandbefab",
                             name="_".join([dname, cname]),
                             tags=['distance',
                                   'metrics'])
            for severity in [1, 3, 5]:
                _logger.info(f"Running {dname} with {cname} and severity {severity}")
                corruption_transform = get_corruption_transform(clip_transform, ccorr, severity)
                dataset = dset(data_path=Config.DATAPATH,
                               split='test',
                               transform=corruption_transform)
                isolated_classes = IsolatedClasses(dataset,
                                                   batch_size=512
                                                   )
                # perform zsoodd
                metrics_dict = run_single_dataset_ood(isolated_classes=isolated_classes,
                                                      clip_model=clip_model,
                                                      clip_tokenizer=clip_tokenizer,
                                                      bert_tokenizer=bert_tokenizer,
                                                      bert_model=bert_model,
                                                      runs=args.runs,
                                                      shorten=args.shorten)
                metrics_dict['severity'] = severity

                wandb.log(metrics_dict)

            run.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--max_splits", type=int)
    parser.add_argument("--split", type=int)
    parser.add_argument("--shorten", type=int, default=0)

    args = parser.parse_args()
    run_all(args)


if __name__ == '__main__':
    main()

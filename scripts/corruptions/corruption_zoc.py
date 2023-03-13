import logging

_logger = logging.getLogger(__name__)



def run_single_dataset_ood(isolated_classes, clip_model, clip_tokenizer, bert_tokenizer, bert_model,
                           id_split, runs):

    from zoc.detectors import zoc_detector


    labels = list(isolated_classes.keys())
    _logger.info(f'Running with classes {labels[:10]} ...')

    metrics = zoc_detector(isolated_classes,
                           clip_model,
                           clip_tokenizer,
                           bert_tokenizer,
                           bert_model,
                           id_split,
                           runs=runs)
    return metrics


def main(args):
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

    _logger.info('Loading decoder model')
    bert_model = get_decoder()

    if args.max_splits !=0:
        splits = np.array_split(list(CorruptionSets.keys()), args.max_splits)
        datasets = splits[args.spllit -1]
    else:
        datasets = CorruptionSets.keys()
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
                transform = get_corruption_transform(clip_transform, ccorr, severity)
                isolated_classes = IsolatedClasses(dataset=dset(data_path=Config.DATAPATH,
                                                                split='test',
                                                                transform=transform),
                                                   batch_size=512
                                                   )
                    # perform zsoodd
                metrics_dict = run_single_dataset_ood(isolated_classes=isolated_classes,
                                                      clip_model=clip_model,
                                                      clip_tokenizer=clip_tokenizer,
                                                      bert_tokenizer=bert_tokenizer,
                                                      bert_model=bert_model,
                                                      id_split=Config.ID_SPLIT,
                                                      runs=args.runs_ood)
                metrics_dict['severity'] = severity

                wandb.log(metrics_dict)

            run.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--max_splits", type=int)
    parser.add_argument("--split", type=int)
    parser.add_argument("--shorten", type=int, default=0)

    args = parser.parse_args()
    main(args)

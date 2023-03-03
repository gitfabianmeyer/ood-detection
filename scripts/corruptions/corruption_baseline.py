

def run_all(args):
    import logging
    _logger = logging.getLogger(__name__)

    import clip
    import numpy as np
    import wandb

    from datasets.corruptions import THESIS_CORRUPTIONS, get_corruption_transform
    from datasets.config import CorruptionSets
    from ood_detection.config import Config
    from zeroshot.utils import get_feature_dict_from_class
    from zoc.detectors import linear_layer_detector

    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)

    if args.max_splits !=0:
        splits = np.array_split(list(CorruptionSets.keys()))
        datasets = splits[args.split]
    else:
        datasets = CorruptionSets.keys()
    for dname in datasets:
        dset = CorruptionSets[dname]
        for cname, ccorr in THESIS_CORRUPTIONS.items():
            run = wandb.init(project="thesis-corruptions-baseline",
                             entity="wandbefab",
                             name="_".join([dname, cname]),
                             tags=['distance',
                                   'metrics'])
            for severity in [1, 3, 5]:
                _logger.info(f"---------------- Running {dname} with {cname} and severity {severity} ---------------")
                transform = get_corruption_transform(clip_transform, ccorr, severity)

                all_features = get_feature_dict_from_class(dset,
                                                           ['train', 'val', 'test'],
                                                           clip_model,
                                                           transform)
                linear_metrics = linear_layer_detector(all_features["train"],
                                                       all_features["val"],
                                                       all_features["test"],
                                                       args.runs,
                                                       Config.ID_SPLIT,
                                                       'logistic')
                linear_metrics["severity"] = severity
                wandb.log(linear_metrics)
            run.finish()


def main(args):
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_all(args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max_splits", type=int, default=0)
    args = parser.parse_args()
    main(args)

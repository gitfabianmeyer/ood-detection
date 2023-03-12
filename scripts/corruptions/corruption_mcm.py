

def run_all(args):
    import logging
    _logger = logging.getLogger(__name__)

    import clip
    import numpy as np
    import wandb

    from datasets.corruptions import THESIS_CORRUPTIONS, get_corruption_transform
    from datasets.config import CorruptionSets
    from ood_detection.config import Config
    from zeroshot.utils import get_feature_and_class_weight_dict_from_dataset

    from zoc.detectors import zeroshot_detector

    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)

    if args.max_splits !=0:
        splits = np.array_split(list(CorruptionSets.keys()), args.max_splits)
        datasets = splits[args.split]
    else:
        datasets = CorruptionSets.keys()
    for dname in datasets:
        dset = CorruptionSets[dname]
        for cname, ccorr in THESIS_CORRUPTIONS.items():
            run = wandb.init(project=f"thesis-corruptions-{args.strategy}-1",
                             entity="wandbefab",
                             name="_".join([dname, cname]),
                             tags=['distance',
                                   'metrics'])
            for severity in [1, 3, 5]:
                _logger.info(f"---------------- Running {dname} with {cname} and severity {severity} ---------------")
                transform = get_corruption_transform(clip_transform, ccorr, severity)

                dataset = dset(Config.DATAPATH,
                               transform=transform,
                               split='test')
                feature_dict, classes_weight_dict = get_feature_and_class_weight_dict_from_dataset(dataset, clip_model)
                use_softmax = True if args.strategy == 'msp' else False
                shorten_classes = None if args.shorten == 0 else args.shorten

                mcm_metrics = zeroshot_detector(feature_dict, classes_weight_dict,
                                                Config.ID_SPLIT, args.runs,
                                                1, use_softmax,
                                                shorten_classes)
                mcm_metrics["severity"] = severity
                wandb.log(mcm_metrics)
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
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--max_splits", type=int)
    parser.add_argument("--split", type=int)

    args = parser.parse_args()
    main(args)

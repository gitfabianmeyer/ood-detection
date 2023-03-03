

def run_all(args):
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"


    import logging
    import clip
    import wandb
    from datasets.config import DATASETS_DICT
    from ood_detection.config import Config
    from zoc.baseline import linear_layer_detector
    from datasets.config import HalfOneDict, HalfTwoDict

    _logger = logging.getLogger(__name__)

    # for each dataset

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)

    if args.split == 1:
        datasets = HalfOneDict
    elif args.split == 2:
        datasets = HalfTwoDict

    else:
        datasets = DATASETS_DICT

    for dname, dset in datasets.items():
        _logger.info(f"---------------Running {dname}--------------")
        run = wandb.init(project=f"thesis-ood_baseline-{args.classifier_type}-full_classes-test_sets",
                         entity="wandbefab",
                         name=dname)

        metrics = linear_layer_detector(args.classifier_type, dset, clip_model, clip_transform,
                                        args.runs) # TODO
        wandb.log(metrics)
        run.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--split', type=int, )
    parser.add_argument('--classifier_type', type=str, required=True)
    args = parser.parse_args()
    run_all(args)


if __name__ == '__main__':
    main()

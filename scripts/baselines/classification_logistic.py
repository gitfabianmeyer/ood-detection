import logging



_logger = logging.getLogger(__name__)


def run_all(args):
    from zeroshot.utils import FeatureSet, FeatureDict
    from zoc.detectors import train_log_reg_classifier
    import numpy as np
    from datasets.config import DATASETS_DICT
    import wandb
    import clip
    from ood_detection.config import Config

    _logger.info(f"Loading {args.vision}")
    clip_model, clip_transform = clip.load(args.vision)
    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")

    for dname in datasets:
        if args.dname:
            if dname != args.dname:
                _logger.info(f"Jumping {dname}")
                continue

        _logger.info(f"\t\t RUNNING {dname}")
        dset = DATASETS_DICT[dname]


        train_set = dset(Config.DATAPATH,
                         transform=clip_transform,
                         split='train')

        train_dict = FeatureDict(train_set,
                                 clip_model)

        train = FeatureSet(train_dict, train_set.classes, train_set.class_to_idx)
        val_dict = FeatureDict(dset(Config.DATAPATH,
                                    transform=clip_transform,
                                    split='val'),
                               clip_model)

        val = FeatureSet(val_dict, train_set.classes, train_set.class_to_idx)
        run = wandb.init(project=f"thesis-classification-logistic",
                         entity="wandbefab",
                         name=dname)
        lr_classifier = train_log_reg_classifier(train,
                                                 val,
                                                 args.cs)

        _logger.info("Getting test acc")
        test_set = dset(Config.DATAPATH,
                        transform=clip_transform,
                        split='test')

        test_dict = FeatureDict(test_set, clip_model)
        test = FeatureSet(test_dict, test_set.classes, test_set.class_to_idx)

        score = lr_classifier.score(test.features.cpu(), test.targets)
        wandb.log({"Acc": score})
        run.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--cs", type=int, default=21)
    parser.add_argument("--dname", type=str, default=None)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max_split", type=int, default=2)
    parser.add_argument("--vision", type=str, default='ViT-B/32')
    args = parser.parse_args()

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_all(args)


if __name__ == '__main__':
    main()

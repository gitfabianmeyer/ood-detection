import logging

from ood_detection.linear import get_test_accuracy_from_dset
from zeroshot.utils import FeatureSet, FeatureDict

_logger = logging.getLogger(__name__)


def run_all(args):
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

        from ood_detection.config import Config
        from adapters.linear import train_classification_head

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

        feature_shape = clip_model.visual.output_dim
        output_shape = len(train_set.classes)

        max_accuracy = 0.
        for learning_rate in np.logspace(np.log2(0.0001), np.log2(0.01), args.lr, base=2):
            run = wandb.init(project=f"thesis-classification-logistic",
                             entity="wandbefab",
                             name=dname)
            lr_acc, lr_classifier = train_classification_head(train,
                                                              val,
                                                              learning_rate,
                                                              args.train_epochs,
                                                              feature_shape,
                                                              output_shape,
                                                              True)
            if lr_acc > max_accuracy:
                max_accuracy = lr_acc
                best_classifier = lr_classifier
            run.finish()

        name = f"thesis-classification-linear_head-{dname}" if args.vision == Config.VISION_MODEL else f"thesis-classification-linear_head-{dname}-{args.vision}"
        run = wandb.init(project=name,
                         entity="wandbefab",
                         name="test",
                         config={'epochs': args.train_epochs,
                                 'lr': args.lr})
        _logger.info("Getting test acc")
        test_set = dset(Config.DATAPATH,
                        transform=clip_transform,
                        split='test')


        test_dict = FeatureDict(test_set, clip_model)
        test = FeatureSet(test_dict, test_set.classes, test_set.class_to_idx)
        acc = get_test_accuracy_from_dset(test, best_classifier)
        wandb.log({"test accuracy": acc})
        run.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--lr", type=int, default=21)
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

import logging

import torch
from zoc.baseline import FeatureSet

_logger = logging.getLogger(__name__)

@torch.no_grad()
def eval_test(dset, classifier):
    classifier.eval()
    print("eval test")
    # test_dict = get_feature_dict(dset(Config.DATAPATH,
    #                                   transform=clip_transform,
    #                                   split='val'),
    #                              clip_model)
    # test = FeatureSet(test_dict, train_set.classes, train_set.class_to_idx)


def run_all(args):
    import numpy as np
    from datasets.config import DATASETS_DICT
    import wandb
    import clip
    from ood_detection.config import Config

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
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
        from zeroshot.utils import get_feature_dict
        train_dict = get_feature_dict(train_set,
                                      clip_model)

        train = FeatureSet(train_dict, train_set.classes, train_set.class_to_idx)
        val_dict = get_feature_dict(dset(Config.DATAPATH,
                                         transform=clip_transform,
                                         split='val'),
                                    clip_model)

        val = FeatureSet(val_dict, train_set.classes, train_set.class_to_idx)


        feature_shape = clip_model.visual.output_dim
        output_shape = len(train_set.classes)

        min_val_loss = np.inf
        for learning_rate in np.logspace(np.log2(0.0001), np.log2(0.1), args.lr, base=2):
            run = wandb.init(project=f"thesis-classification-linear_head-{dname}",
                             entity="wandbefab",
                             name=str(learning_rate),
                             config={'epochs': args.train_epochs,
                                     'lr': args.lr})
            results = train_classification_head(train,
                                                val,
                                                None,
                                                learning_rate,
                                                args.train_epochs,
                                                feature_shape,
                                                output_shape,
                                                True)
            if results["min loss"] < min_val_loss:
                min_val_loss = results["min loss"]
                best_result = results

            eval_test()


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

    args = parser.parse_args()

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_all(args)


if __name__ == '__main__':
    main()

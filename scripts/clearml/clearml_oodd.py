import numpy as np
from zeroshot.utils import get_feature_dict_from_class


def run_all(args):
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    import logging
    import clip
    import wandb
    from ood_detection.config import Config
    from zoc.baseline import linear_layer_detector
    from datasets.config import DATASETS_DICT

    _logger = logging.getLogger(__name__)

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

        _logger.info(f"---------------Running {dname}--------------")
        run = wandb.init(project=f"TEST-thesis-ood-benchmark-{args.classifier_type}",
                         entity="wandbefab",
                         name=dname,
                         config={'type': args.classifier_type,
                                 'lr': args.lr})

        dset = DATASETS_DICT[dname]
        all_features = get_feature_dict_from_class(dset,
                                                   ['train', 'val', 'test'],
                                                   clip_model,
                                                   clip_transform)
        train = all_features['train']
        val = all_features["val"]
        test = all_features["test"]

        metrics = linear_layer_detector(train, val, test,
                                        args.runs,
                                        Config.ID_SPLIT,
                                        args.classifier_type,
                                        epochs=300,
                                        learning_rate=0.001)
        wandb.log(metrics)
        run.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument("--dname", type=str, default=None)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max_split", type=int, default=0)
    parser.add_argument('--classifier_type', type=str, required=True)
    # parser.add_argument("--vision", type=str, default='ViT-L/14@336px')
    parser.add_argument("--vision", type=str, default='ViT-B/32')
    parser.add_argument("--epochs", type=int, default=300)

    args = parser.parse_args()
    run_all(args)


if __name__ == '__main__':
    main()

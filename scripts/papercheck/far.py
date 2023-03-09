import logging


_logger = logging.getLogger(__name__)


def main():
    import clip
    import numpy as np
    import wandb
    from datasets.config import DATASETS_DICT
    from ood_detection.classification_utils import zeroshot_classifier
    from ood_detection.config import Config
    from zeroshot.utils import get_set_features_no_classes
    from zoc.ablation import get_clip_auroc_from_features
    from datasets.classnames import base_template

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    temperatures = np.logspace(np.log2(0.001), np.log2(1000), num=args.temps, base=2.0)

    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")

    for id_name in datasets:
        id_set = DATASETS_DICT[id_name]
        id_dataset = id_set(Config.DATAPATH,
                            transform=clip_transform,
                            split='test')

        id_set_features = get_set_features_no_classes(id_dataset, clip_model)
        zeroshot_weights = zeroshot_classifier(id_dataset.classes, base_template, clip_model)
        for od_name, od_set in DATASETS_DICT.items():
            if od_name == id_name:
                continue
            run = wandb.init(project=f"thesis-far-{id_name}",
                             entity="wandbefab",
                             name=od_name)
            ood_dataset = od_set(Config.DATAPATH,
                                 transform=clip_transform,
                                 split='test')

            ood_set_features = get_set_features_no_classes(ood_dataset, clip_model)
            for temp in temperatures:
                auroc_clip_score = get_clip_auroc_from_features(id_set_features, ood_set_features, zeroshot_weights,
                                                                temp)
                wandb.log({'MCM AUROC': auroc_clip_score,
                           'temperature': temp})
            run.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument("--temps", type=int, default=50)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max_split", type=int, default=0)
    args = parser.parse_args()

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main()

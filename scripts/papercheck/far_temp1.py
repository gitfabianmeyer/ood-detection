import logging

_logger = logging.getLogger(__name__)


def main():
    import clip
    import numpy as np
    import wandb
    from tqdm import tqdm

    from datasets.config import DATASETS_DICT
    from ood_detection.classification_utils import zeroshot_classifier
    from ood_detection.config import Config
    from zeroshot.utils import get_set_features_no_classes
    from zoc.ablation import get_clip_auroc_from_features
    from datasets.classnames import base_template

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    datasets = DATASETS_DICT.keys()
    features = {}
    classifiers = {}
    _logger.info('generating all features')
    for name in tqdm(datasets):
        _logger.info(name)
        d = DATASETS_DICT[name]
        dset = d(Config.DATAPATH,
                 transform=clip_transform,
                 split='test')
        features[name] = get_set_features_no_classes(dset, clip_model)
        classifiers[name] = zeroshot_classifier(dset.classes, base_template, clip_model)

    for temp in args.temps:
        for id_name in datasets:
            id_set_features = features[id_name]
            zeroshot_weights = classifiers[id_name]
            for od_name in datasets:
                run = wandb.init(project=f"thesis-far-ood-msp-{str(args.temp)}",
                                 entity="wandbefab",
                                 name=id_name + "-" + od_name)
                _logger.info(f"Running {id_name} vs {od_name}")
                if od_name == id_name:
                    continue
                ood_set_features = features[od_name]
                auroc_clip_score = get_clip_auroc_from_features(id_set_features, ood_set_features, zeroshot_weights, temp)
                wandb.log({'AUROC': auroc_clip_score})
                run.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('-t', '--temps', help='delimited list input',
                        type=lambda s: [float(item) for item in s.split(',')])
    args = parser.parse_args()

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main()

import logging

_logger = logging.getLogger(__name__)


def main(args):
    import clip
    import wandb
    from tqdm import tqdm

    from datasets.config import DATASETS_DICT
    from ood_detection.config import Config
    from zeroshot.utils import get_feature_and_classifier_dict_for_datasets
    from zoc.ablation import get_clip_auroc_from_features
    from datasets.classnames import base_template

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    datasets = DATASETS_DICT.keys()
    features, classifiers = get_feature_and_classifier_dict_for_datasets(datasets, clip_model,
                                                                         clip_transform,
                                                                         split='test',
                                                                         template=base_template)
    for temp in args.temps:
        _logger.info(f'temperature: {temp}')
        for id_name in datasets:
            id_set_features = features[id_name]
            zeroshot_weights = classifiers[id_name]
            for od_name in tqdm(datasets):
                if od_name == id_name:
                    _logger.info(f"Cant run against myself: {id_name}vs {od_name}")
                    continue
                run = wandb.init(project=f"thesis-far-ood-{args.strategy}-{str(temp)}",
                                 entity="wandbefab",
                                 name=id_name + "-" + od_name,
                                 config=args.__dict__
                                 )
                _logger.info(f"Running {id_name} vs {od_name}")

                ood_set_features = features[od_name]
                auroc_clip_score = get_clip_auroc_from_features(id_set_features, ood_set_features, zeroshot_weights,
                                                                temp, args.strategy)
                wandb.log({'AUROC': auroc_clip_score})
                run.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('-t', '--temps', help='delimited list input',
                        type=lambda s: [float(item) for item in s.split(',')])
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--runs', type=int, default=10)

    args = parser.parse_args()
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)
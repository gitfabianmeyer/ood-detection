import logging

_logger = logging.getLogger(__name__)


def main(args):
    import clip
    import wandb
    from datasets.config import DATASETS_DICT
    from zeroshot.utils import get_feature_and_class_weight_dict_from_dataset
    from zoc.detectors import zeroshot_detector
    from ood_detection.config import Config
    from datasets.classnames import base_template

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    datasets = DATASETS_DICT.keys()

    for dname in datasets:
        _logger.info(f"Running {dname}")

        dset = DATASETS_DICT[dname]
        dataset = dset(Config.DATAPATH,
                       transform=clip_transform,
                       split='test')
        dataset.templates = base_template
        feature_dict, classes_weight_dict = get_feature_and_class_weight_dict_from_dataset(dset,
                                                                                           clip_model)
        for temp in args.temps:
            _logger.info(f'temperature: {temp}')
            use_softmax = True if args.strategy == 'msp' else 'mls'
            shorten_classes = None if args.shorten == 0 else args.shorten
            project_name = f"thesis-near-mcm-{args.strategy}-{temp}"
            run = wandb.init(project=project_name,
                             entity="wandbefab",
                             name=dname,
                             config=args.__dict__)

            metrics = zeroshot_detector(feature_dict, classes_weight_dict,
                                        Config.ID_SPLIT, args.runs,
                                        temp, use_softmax,
                                        shorten_classes)
            wandb.log(metrics)
            run.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('-t', '--temps', help='delimited list input',
                        type=lambda s: [float(item) for item in s.split(',')])
    parser.add_argument('--shorten', type=int, default=0)
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--runs', type=int, default=10)

    args = parser.parse_args()
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main()

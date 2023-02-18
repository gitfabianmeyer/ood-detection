def run_all(args):
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    from zoc.baseline import linear_layer_detector
    import logging
    import clip
    import wandb
    from datasets.config import DATASETS_DICT
    from ood_detection.config import Config

    _logger = logging.getLogger(__name__)

    # for each dataset

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    for dname, dset in DATASETS_DICT.items():
        _logger.info(f"---------------Running {dname}--------------")
        # run = wandb.init(project=f"thesis-ood_baseline_{args.classifier_type}_full_classes_test_sets",
        #                  entity="wandbefab",
        #                  name=dname)

        metrics = linear_layer_detector(args.classifier_type, dset, clip_model, clip_transform,
                                        args.runs)
        print(metrics)
        # wandb.log(metrics)
        # run.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--split', type=int, )
    parser.add_argument('--classifier_type', type=str, required=True)
    args = parser.parse_args()
    run_all(args)


if __name__ == '__main__':
    main()

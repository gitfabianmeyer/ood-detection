import logging

_logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_all()


def run_all():
    import clip
    from datasets.config import DATASETS_DICT
    import wandb
    from ood_detection.config import Config

    clip_model, clip_transfrom = clip.load(Config.VISION_MODEL)

    for dname, dset in DATASETS_DICT.items():
        run = wandb.init(project=f"thesis-logreg-classification",
                         entity="wandbefab",
                         name=dname)
        _logger.info(dname)
        result = logreg_baseline(dset, clip_model, clip_transfrom)
        wandb.log(result)

        run.finish()


def logreg_baseline(dataset, clip_model, clip_transform):
    from datasets.zoc_loader import IsolatedClasses
    from ood_detection.config import Config
    from zoc.baseline import get_feature_weight_dict, FeatureSet, train_log_reg_classifier

    train_dataset = dataset(Config.DATAPATH,
                            split='train',
                            transform=clip_transform)

    isolated_classes = IsolatedClasses(train_dataset,
                                       batch_size=512)
    feature_weight_dict_train = get_feature_weight_dict(isolated_classes, clip_model)

    isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                               split='val',
                                               transform=clip_transform),
                                       batch_size=512)
    feature_weight_dict_val = get_feature_weight_dict(isolated_classes, clip_model)

    # train classifier to classify id set
    train_set = FeatureSet(feature_weight_dict_train, train_dataset.classes, train_dataset.class_to_idx)
    val_set = FeatureSet(feature_weight_dict_val, train_dataset.classes, train_dataset.class_to_idx)
    logistic_classifier = train_log_reg_classifier(train_set, val_set, 96)

    isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                               split='test',
                                               transform=clip_transform),
                                       batch_size=512)
    feature_weight_dict_test = get_feature_weight_dict(isolated_classes, clip_model)
    test_set = FeatureSet(feature_weight_dict_test, train_dataset.classes, train_dataset.class_to_idx)

    score = logistic_classifier.score(test_set.features.cpu(), test_set.targets.cpu())
    return {"Acc": score,
            "C": logistic_classifier.C}


if __name__ == '__main__':
    main()

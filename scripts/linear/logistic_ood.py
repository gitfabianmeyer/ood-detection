import logging

import numpy as np

_logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--max_split", type=int)
    args = parser.parse_args()
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_all(args)


def run_all(args):
    import wandb
    import clip
    from ood_detection.config import Config
    from datasets.config import DATASETS_DICT

    clip_model, clip_transfrom = clip.load(Config.VISION_MODEL)

    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")

    for dname in datasets:
        _logger.info(f"\t\t RUNNING {dname}")
        dset = DATASETS_DICT[dname]
        run = wandb.init(project=f"thesis-logreg-ood-{args.runs}_runs",
                         entity="wandbefab",
                         name=dname)
        _logger.info(dname)
        result = logred_oodd(dset, clip_model, clip_transfrom, Config.ID_SPLIT, args.runs)
        wandb.log(result)
        run.finish()


def logred_oodd(dataset, clip_model, clip_transform, id_split, runs):
    from ood_detection.config import Config
    from tqdm import tqdm
    import numpy as np
    import torch
    from datasets.zoc_loader import IsolatedClasses
    from zoc.baseline import get_feature_weight_dict, FeatureSet, train_log_reg_classifier
    from zoc.utils import get_ablation_splits, get_split_specific_targets, get_auroc_for_max_probs
    train_dataset = dataset(Config.DATAPATH,
                            split='train',
                            transform=clip_transform)

    id_classes = int(id_split * len(train_dataset.classes))
    ood_classes = len(train_dataset.classes) - id_classes
    isolated_classes = IsolatedClasses(train_dataset,
                                       batch_size=512)
    feature_weight_dict_train = get_feature_weight_dict(isolated_classes, clip_model)

    isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                               split='val',
                                               transform=clip_transform),
                                       batch_size=512)
    feature_weight_dict_val = get_feature_weight_dict(isolated_classes, clip_model)
    isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                               split='test',
                                               transform=clip_transform),
                                       batch_size=512)
    feature_weight_dict_test = get_feature_weight_dict(isolated_classes, clip_model)

    ablation_splits = get_ablation_splits(isolated_classes.classes, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    logistic_aucs = []
    for ablation_split in tqdm(ablation_splits):

        class_to_idx_mapping = {label: i for (i, label) in enumerate(ablation_split)}
        seen_labels = ablation_split[:id_classes]
        unseen_labels = ablation_split[id_classes:]
        _logger.info(f"Seen labels: {seen_labels}\nOOD Labels: {unseen_labels}")

        # train classifier to classify id set
        train_set = FeatureSet(feature_weight_dict_train, seen_labels, class_to_idx_mapping)
        val_set = FeatureSet(feature_weight_dict_val, seen_labels, class_to_idx_mapping)

        logistic_classifier = train_log_reg_classifier(train_set, val_set, 96)
        # eval for ood detection

        logreg_probs_max = []
        for i, semantic_label in enumerate(ablation_split):
            # get features
            image_features_for_label = feature_weight_dict_test[semantic_label]
            # calc the logits and softmaxs
            logistic_logits = logistic_classifier.predict_proba(image_features_for_label.cpu())
            top_log_logits, _ = torch.topk(torch.Tensor(logistic_logits), 1, -1)

            # detection score is accumulative sum of probs of generated entities
            # careful, only for this setting axis=1
            logreg_probs_max.extend(top_log_logits.detach().numpy())

        targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)

        logistic_aucs.append(get_auroc_for_max_probs(targets, logreg_probs_max))

    result = {"logreg": np.mean(logistic_aucs), "logreg_std": np.std(logistic_aucs)}

    return result


if __name__ == '__main__':
    main()

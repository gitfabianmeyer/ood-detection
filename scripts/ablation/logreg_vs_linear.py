import logging

import clip
import numpy as np
import torch
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config
from zoc.baseline import get_feature_weight_dict, FeatureSet
from zoc.utils import get_ablation_splits, get_split_specific_targets, fill_auc_lists, fill_f_acc_lists, \
    get_result_mean_dict

from src.datasets.config import DATASETS_DICT
from src.zoc.baseline import train_id_classifier, train_log_reg_classifier
from src.zoc.utils import get_auroc_for_max_probs

_logger = logging.getLogger(__name__)


def main():
    clip_model, clip_transfrom = clip.load(Config.VISION_MODEL)

    for dname, dset in DATASETS_DICT.items():
        _logger.info(dname)
        log_reg_stuff(dset, clip_model, clip_transfrom, 10, 1)


def log_reg_stuff(dataset, clip_model, clip_transform, id_classes, runs):
    device = Config.DEVICE
    train_dataset = dataset(Config.DATAPATH,
                            split='train',
                            transform=clip_transform)

    ood_classes = len(train_dataset.classes) - id_classes
    isolated_classes = IsolatedClasses(train_dataset,
                                       batch_size=512)
    # feature_weight_dict_train = get_feature_weight_dict(isolated_classes, clip_model, device)

    classes = ["1", "2", "3", "4", "5"]
    feature_weight_dict_train = {key: torch.rand((5,10)) for key in classes}
    isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                               split='val',
                                               transform=clip_transform),
                                       batch_size=512)
    # feature_weight_dict_val = get_feature_weight_dict(isolated_classes, clip_model, device)
    feature_weight_dict_val = {key: torch.rand((5, 10)) for key in classes}
    ablation_splits = get_ablation_splits(isolated_classes.classes, n=1, id_classes=id_classes, # TODO
                                          ood_classes=ood_classes)

    logistic_aucs, linear_aucs = [], []
    for ablation_split in ablation_splits:

        # class_to_idx_mapping = {label: i for (i, label) in enumerate(ablation_split)}
        # seen_labels = ablation_split[:id_classes]
        # unseen_labels = ablation_split[id_classes:]
        seen_labels = classes[:3]
        unseen_labels = classes[3:]
        class_to_idx_mapping = {key:i for (i, key) in enumerate(classes)}
        _logger.info(f"Seen labels: {seen_labels}\nOOD Labels: {unseen_labels}")

        # train classifier to classify id set
        train_set = FeatureSet(feature_weight_dict_train, seen_labels, class_to_idx_mapping)
        val_set = FeatureSet(feature_weight_dict_val, seen_labels, class_to_idx_mapping)


        logistic_classifier = train_log_reg_classifier(train_set, val_set)
        linear_classifier = train_id_classifier(train_set, val_set, wandb_logging=False)
        # eval for ood detection

        isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                                   split='test',
                                                   transform=clip_transform),
                                           batch_size=512)
        # feature_weight_dict_test = get_feature_weight_dict(isolated_classes, clip_model, device) # TODO
        feature_weight_dict_test = {key: torch.rand((5, 10)) for key in classes}

        logreg_probs_max, linear_probs_max = [], []
        for i, semantic_label in enumerate(ablation_split):
            # get features
            image_features_for_label = feature_weight_dict_test[semantic_label]
            # calc the logits and softmaxs
            linear_logits = linear_classifier(image_features_for_label.to(torch.float32).to(device))
            top_prob, _ = linear_logits.cpu().topk(1, dim=-1)
            _, logistic_logits = torch.topk(torch.Tensor(logistic_classifier.predict_proba(image_features_for_label)),
                                            1, 1)

            assert linear_logits.shape[1] == id_classes
            assert logistic_logits.shape[1] == id_classes

            # detection score is accumulative sum of probs of generated entities
            # careful, only for this setting axis=1
            linear_probs_max.extend(top_prob.detach().numpy())
            logreg_probs_max.extend(top_prob.detach().numpy())

        targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)

        linear_aucs.append(get_auroc_for_max_probs(targets, linear_probs_max))
        logistic_aucs.append(get_auroc_for_max_probs(targets, logreg_probs_max))


    print(f" linear: {np.mean(linear_aucs)}")
    print(f" logistic: {np.mean(logistic_aucs)}")
    return True

if __name__=='__main__':
    main()
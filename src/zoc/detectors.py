import logging
import random

import numpy as np
import torch
from datasets.zoc_loader import IsolatedClasses
from zeroshot.classification import get_cosine_similarity_matrix_for_normed_features
from ood_detection.config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ood_detection.ood_utils import sorted_zeroshot_weights
from zeroshot.utils import FeatureDict, FeatureSet
from zoc.utils import fill_auc_lists, fill_f_acc_lists, get_result_mean_dict, get_auroc_for_max_probs, get_mean_std, \
    get_ablation_splits, get_split_specific_targets, get_mean_max_sum_for_zoc_image

_logger = logging.getLogger(__name__)


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


# for testing purpose
def get_fake_dict(isolated_classe):
    return {c: torch.rand((random.randint(1, 20), 5)) for c in isolated_classe.classes}


@torch.no_grad()
def zeroshot_detector(feature_weight_dict: FeatureDict,
                      classes_weight_dict: FeatureDict,
                      id_classes,
                      ood_classes,
                      runs,
                      temperature=1):
    ablation_splits = get_ablation_splits(feature_weight_dict.classes, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    metrics_list = []
    # for each temperature..
    auc_list_sum, auc_list_mean, auc_list_max = [], [], []
    for split in ablation_splits:

        seen_labels = split[:id_classes]
        unseen_labels = split[id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")

        zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)

        ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
        f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

        for i, semantic_label in enumerate(split):
            # get features
            image_features_for_label = feature_weight_dict[semantic_label]
            # calc the logits and softmaxs
            zeroshot_probs = get_cosine_similarity_matrix_for_normed_features(image_features_for_label,
                                                                              zeroshot_weights,
                                                                              1)
            assert zeroshot_probs.shape[1] == id_classes
            # detection score is accumulative sum of probs of generated entities
            # careful, only for this setting axis=1
            ood_prob_sum = np.sum(zeroshot_probs.detach().cpu().numpy(), axis=1)
            ood_probs_sum.extend(ood_prob_sum)

            ood_prob_mean = np.mean(zeroshot_probs.detach().cpu().numpy(), axis=1)
            ood_probs_mean.extend(ood_prob_mean)

            top_prob, _ = zeroshot_probs.cpu().topk(1, dim=-1)
            ood_probs_max.extend(top_prob.detach().numpy())

            id_probs_sum.extend(1. - ood_prob_sum)

        targets = get_split_specific_targets(feature_weight_dict, seen_labels, unseen_labels)
        fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                       targets)
        fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

    metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)
    metrics["temperature"] = temperature

    metrics_list.append(metrics)

    return metrics_list


@torch.no_grad()
def zoc_detector(isolated_classes: IsolatedClasses,
                 clip_model,
                 clip_tokenizer,
                 bert_tokenizer,
                 bert_model,
                 id_split,
                 runs,
                 shorten_classes=None):
    all_classes = isolated_classes.classes
    device = Config.DEVICE
    if not shorten_classes:
        id_classes = len(isolated_classes.classes) * id_split
        ood_classes = len(isolated_classes.classes) - id_classes


    else:
        _logger.warning(f"SHORTENING CLASSES TO {shorten_classes}")

        id_classes = int(shorten_classes * Config.ID_SPLIT)
        ood_classes = shorten_classes - id_classes

    ablation_splits = get_ablation_splits(all_classes, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum, auc_list_mean, auc_list_max = [], [], []
    for split in ablation_splits:
        seen_labels = split[:id_classes]
        unseen_labels = split[id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

        ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
        for i, semantic_label in enumerate(split):
            loader = isolated_classes[semantic_label]
            for idx, image in enumerate(tqdm(loader)):
                clip_out = clip_model.encode_image(image.to(device)).float()
                ood_prob_max, ood_prob_mean, ood_prob_sum = get_mean_max_sum_for_zoc_image(bert_model, bert_tokenizer,
                                                                                           clip_model, clip_tokenizer,
                                                                                           device, id_classes, clip_out,
                                                                                           seen_descriptions,
                                                                                           seen_labels)

                ood_probs_sum.append(ood_prob_sum)
                ood_probs_mean.append(ood_prob_mean)
                ood_probs_max.append(ood_prob_max.detach().numpy())
                id_probs_sum.append(1 - ood_prob_sum)

        targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)
        fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                       targets)
        fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

    metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)

    return metrics


@torch.no_grad()
def zoc_detector_featuredict(feature_dict,
                             clip_model,
                             clip_tokenizer,
                             bert_tokenizer,
                             bert_model,
                             device,
                             id_split,
                             runs,
                             shorten_classes=None):
    if shorten_classes:
        id_classes = int(shorten_classes * id_split)
        ood_classes = shorten_classes - id_classes

    else:
        id_classes = int(len(feature_dict.keys)) * id_split
        ood_classes = len(feature_dict.keys) - id_classes
    ablation_splits = get_ablation_splits(feature_dict.keys(), n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum, auc_list_mean, auc_list_max = [], [], []
    for split in ablation_splits:
        seen_labels = split[:id_classes]
        unseen_labels = split[id_classes:]
        _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

        ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
        f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

        for semantic_label in split:
            image_features = feature_dict[semantic_label]
            for image in tqdm(image_features):
                ood_prob_max, ood_prob_mean, ood_prob_sum = get_mean_max_sum_for_zoc_image(bert_model, bert_tokenizer,
                                                                                           clip_model, clip_tokenizer,
                                                                                           device, id_classes, image,
                                                                                           seen_descriptions,
                                                                                           seen_labels)

                ood_probs_sum.append(ood_prob_sum)
                ood_probs_mean.append(ood_prob_mean)
                ood_probs_max.append(ood_prob_max.detach().numpy())

        targets = get_split_specific_targets(feature_dict, seen_labels, unseen_labels)
        fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                       targets)
        fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

    metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)

    return metrics


def train_linear_id_classifier(train_set: FeatureSet, eval_set: FeatureSet, epochs=10, learning_rate=0.001):
    device = Config.DEVICE
    classifier = LinearClassifier(train_set.features_dim, len(train_set.labels)).to(device)

    train_loader = DataLoader(train_set,
                              batch_size=128,
                              shuffle=True)
    eval_loader = DataLoader(eval_set,
                             batch_size=512,
                             shuffle=True)

    early_stopping = 0
    max_epoch_without_improvement = 3
    optimizer = AdamW(params=classifier.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    best_val_loss = np.inf
    for epoch in tqdm(range(1, epochs + 1)):

        epoch_loss = 0.
        # train
        for image_features, targets in tqdm(train_loader):
            image_features = image_features.to(torch.float32).to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            preds = classifier(image_features.to(torch.float32).to(device))
            output = criterion(preds, targets)
            output.backward()

            loss = output.detach().item()
            epoch_loss += loss

            optimizer.step()

        # eval
        epoch_val_loss = 0.
        eval_accs = []
        for eval_features, eval_targets in tqdm(eval_loader):

            eval_features = eval_features.to(torch.float32).to(device)
            eval_targets = eval_targets.to(device)

            with torch.no_grad():
                eval_preds = classifier(eval_features)
                eval_loss = criterion(eval_preds, eval_targets).detach().item()

            epoch_val_loss += eval_loss

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_classifier = classifier
            else:
                early_stopping += 1
                _logger.info(f"No improvement on val loss ( {early_stopping} / {max_epoch_without_improvement}")
                if early_stopping == max_epoch_without_improvement + 1:
                    _logger.info(F"Hit the maximum epoch without improvement {max_epoch_without_improvement}. Exiting")
                    return best_classifier

            _, indices = torch.topk(torch.softmax(eval_preds, dim=-1), k=1)
            eval_accs.append(accuracy_score(eval_targets.to('cpu').numpy(), indices.to('cpu').numpy()))

        _logger.info(f"Epoch {epoch} Eval Acc: {np.mean(eval_accs)}")

    return best_classifier


def train_log_reg_classifier(train_set, eval_set, num_cs):
    _logger.info(f"Training logistic regression for {num_cs} Cs....")
    if train_set.features.is_cuda:
        train_set.features = train_set.features.cpu()
    if eval_set.features.is_cuda:
        eval_set.features = eval_set.features.cpu()

    cs = np.logspace(np.log2(0.000001), np.log2(1000000), num_cs, base=2)
    best_classifier = None
    best_score = 0
    for c in cs:
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs', C=c, penalty='l2')
        classifier.fit(train_set.features, train_set.targets)
        score = classifier.score(eval_set.features, eval_set.targets)
        if score > best_score:
            best_classifier = classifier
            best_score = score
            _logger.info(f"New best score: {score:.3f} for {classifier.C}")
    return best_classifier


def linear_layer_detector(train_feature_dict,
                          eval_feature_dict,
                          test_feature_dict,
                          runs,
                          id_classes_split,
                          classifier_type,
                          epochs=300,
                          num_cs=96,
                          learning_rate=0.001):
    assert classifier_type in ['linear', 'logistic', 'all']

    device = Config.DEVICE
    all_classes = test_feature_dict.classes
    id_classes = int(len(all_classes) * id_classes_split)
    ood_classes = len(all_classes) - id_classes

    ablation_splits = get_ablation_splits(all_classes, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_max = []
    auc_list_max_log = []
    for ablation_split in ablation_splits:

        class_to_idx_mapping = {label: i for (i, label) in enumerate(ablation_split)}
        seen_labels = ablation_split[:id_classes]
        unseen_labels = ablation_split[id_classes:]

        # train classifier to classify id set

        train_set = FeatureSet(train_feature_dict, seen_labels, class_to_idx_mapping)
        val_set = FeatureSet(eval_feature_dict, seen_labels, class_to_idx_mapping)

        if classifier_type == 'linear':
            classifier = train_linear_id_classifier(train_set, val_set, epochs, learning_rate)

        elif classifier_type == 'logistic':
            classifier = train_log_reg_classifier(train_set, val_set, num_cs)

        else:
            lin_classifier = train_linear_id_classifier(train_set, val_set, epochs, learning_rate)
            log_classifier = train_log_reg_classifier(train_set, val_set, num_cs)

        ood_probs_max = []
        log_ood_probs_max = []
        for i, semantic_label in enumerate(ablation_split):
            # get features
            image_features_for_label = test_feature_dict[semantic_label]
            # calc the logits and softmaxs
            if classifier_type == 'linear':
                logits = classifier(image_features_for_label.to(torch.float32).to(device))
                top_prob, _ = logits.cpu().topk(1, dim=-1)
                ood_probs_max.extend(top_prob.detach().numpy())
            elif classifier_type == 'logistic':
                logits = classifier.predict_proba(image_features_for_label.cpu())
                top_prob = np.amax(logits, axis=1)
                ood_probs_max.extend(top_prob)

            else:
                lin_logits = lin_classifier(image_features_for_label.to(torch.float32).to(device))
                lin_top_prob, _ = lin_logits.cpu().topk(1, dim=-1)

                log_logits = log_classifier.predict_proba(image_features_for_label.cpu())
                top_log_prob = np.amax(log_logits, axis=1)
                ood_probs_max.extend(lin_top_prob.detach().numpy())
                log_ood_probs_max.extend(top_log_prob)

        targets = get_split_specific_targets(test_feature_dict, seen_labels, unseen_labels)

        if classifier_type != 'all':
            auc_list_max.append(get_auroc_for_max_probs(targets, ood_probs_max))
        else:
            auc_list_max.append(get_auroc_for_max_probs(targets, ood_probs_max))
            auc_list_max_log.append(get_auroc_for_max_probs(targets, log_ood_probs_max))

    if classifier_type != 'all':
        mean, std = get_mean_std(auc_list_max)
        metrics = {'AUC': mean,
                   'std': std}
    else:
        lin_mean, lin_std = get_mean_std(auc_list_max)
        log_mean, log_std = get_mean_std(auc_list_max_log)
        metrics = {'log_AUC': log_mean,
                   'log_std': log_std,
                   'lin_AUC': lin_mean,
                   'lin_std': lin_std}
    return metrics

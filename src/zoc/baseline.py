import logging

import numpy as np
import torch
from zeroshot.classification import get_cosine_similarity_matrix_for_normed_features
from datasets.zoc_loader import IsolatedClasses
from ood_detection.classification_utils import zeroshot_classifier
from ood_detection.config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from zeroshot.classification import get_image_features_for_isolated_class_loader
from zoc.utils import get_ablation_splits, get_split_specific_targets, fill_auc_lists, fill_f_acc_lists, \
    get_result_mean_dict

from ood_detection.ood_utils import sorted_zeroshot_weights

_logger = logging.getLogger(__name__)


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


@torch.no_grad()
def get_feature_weight_dict(isolated_classes, clip_model, device):
    weights_dict = {}
    for cls in isolated_classes.classes:
        loader = isolated_classes[cls]

        features = get_image_features_for_isolated_class_loader(loader, clip_model)
        weights_dict[cls] = features.half()

    return weights_dict


def get_zeroshot_weight_dict(isolated_classes, clip_model):
    weights_dict = {}

    if isinstance(isolated_classes, IsolatedClasses):
        weights = zeroshot_classifier(isolated_classes.classes, isolated_classes.templates, clip_model)

        for classname, weight in zip(isolated_classes.classes, weights):
            weights_dict[classname] = weight

    return weights_dict


@torch.no_grad()
def baseline_detector(clip_model,
                      device,
                      isolated_classes: IsolatedClasses,
                      id_classes,
                      ood_classes,
                      runs,
                      temperature):
    feature_weight_dict = get_feature_weight_dict(isolated_classes, clip_model, device)
    classes_weight_dict = get_zeroshot_weight_dict(isolated_classes, clip_model)

    ablation_splits = get_ablation_splits(isolated_classes.classes, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    metrics_list = []
    # for each temperature..
    for temperature in np.logspace(-7.158429362604483, 6.643856189774724, num=15,
                                   base=2.0):  # 10 values between .007 and 100

        auc_list_sum, auc_list_mean, auc_list_max = [], [], []
        for split in ablation_splits:

            seen_labels = split[:id_classes]
            unseen_labels = split[id_classes:]
            _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")

            zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)

            ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
            f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

            # do 10 times
            for i, semantic_label in enumerate(split):
                # get features
                image_features_for_label = feature_weight_dict[semantic_label]
                # calc the logits and softmaxs
                zeroshot_probs = get_cosine_similarity_matrix_for_normed_features(image_features_for_label,
                                                                                  zeroshot_weights,
                                                                                  temperature)
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

            targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)
            fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                           targets)
            fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

        metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)
        metrics["temperature"] = temperature

        metrics_list.append(metrics)

    return metrics_list


def train_linear_id_classifier(train_set, eval_set, epochs=10, learning_rate=0.001):
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


def train_log_reg_classifier(train_set, eval_set, max_iter=110, cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    if train_set.features.is_cuda:
        train_set.features = train_set.features.cpu()
    print(train_set.features.shape)
    print(len(train_set.labels))
    if eval_set.features.is_cuda:
        eval_set.features = eval_set.features.cpu()
    print(eval_set.features.shape)
    print(len(eval_set.labels))

    best_classifier = None
    best_score = 0
    for c in cs:
        classifier = LogisticRegression(max_iter=max_iter, solver='sag', tol=0.0001, C=c, penalty='l2')
        classifier.fit(train_set.features, train_set.targets)
        score = classifier.score(eval_set.features, eval_set.targets)
        print(f"Eval score ( c: {c}) for log reg: {score}")
        if score > best_score:
            best_classifier = classifier
            best_score = score
    return best_classifier


def linear_layer_detector(classifier_type, dataset, clip_model, clip_transform, runs):
    assert classifier_type in ['linear', 'logistic']
    device = Config.DEVICE
    train_dataset = dataset(Config.DATAPATH,
                            split='train',
                            transform=clip_transform)
    labels = train_dataset.classes
    id_classes = int(len(labels) * Config.ID_SPLIT)
    ood_classes = len(labels) - id_classes

    isolated_classes = IsolatedClasses(train_dataset,
                                       batch_size=512)
    feature_weight_dict_train = get_feature_weight_dict(isolated_classes, clip_model, device)

    isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                               split='val',
                                               transform=clip_transform),
                                       batch_size=512)
    feature_weight_dict_val = get_feature_weight_dict(isolated_classes, clip_model, device)
    ablation_splits = get_ablation_splits(isolated_classes.classes, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    auc_list_sum, auc_list_mean, auc_list_max = [], [], []
    for ablation_split in ablation_splits:

        class_to_idx_mapping = {label: i for (i, label) in enumerate(ablation_split)}
        seen_labels = ablation_split[:id_classes]
        unseen_labels = ablation_split[id_classes:]

        # train classifier to classify id set
        train_set = FeatureSet(feature_weight_dict_train, seen_labels, class_to_idx_mapping)
        val_set = FeatureSet(feature_weight_dict_val, seen_labels, class_to_idx_mapping)

        if classifier_type == 'linear':
            classifier = train_linear_id_classifier(train_set, val_set)

        else:
            classifier = train_log_reg_classifier(train_set, val_set)

        isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                                   split='test',
                                                   transform=clip_transform),
                                           batch_size=512)
        feature_weight_dict_test = get_feature_weight_dict(isolated_classes, clip_model, device)

        ood_probs_sum, ood_probs_mean, ood_probs_max = [], [], []
        f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []
        for i, semantic_label in enumerate(ablation_split):
            # get features
            image_features_for_label = feature_weight_dict_test[semantic_label]
            # calc the logits and softmaxs
            if classifier_type == 'linear':
                logits = classifier(image_features_for_label.to(torch.float32).to(device))
            else:
                logits = classifier(image_features_for_label.cpu())

            top_prob, _ = logits.cpu().topk(1, dim=-1)
            ood_probs_max.extend(top_prob.detach().numpy())

        targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)
        fill_auc_lists(auc_list_max, auc_list_mean, auc_list_sum, ood_probs_mean, ood_probs_max, ood_probs_sum,
                       targets)
        fill_f_acc_lists(acc_probs_sum, f_probs_sum, id_probs_sum, ood_probs_sum, targets)

    metrics = get_result_mean_dict(acc_probs_sum, auc_list_max, auc_list_mean, auc_list_sum, f_probs_sum)

    return metrics


class FeatureSet(Dataset):
    def __init__(self, feature_dict, labels, class_to_idx_mapping):
        self.labels = labels
        self.class_to_idx_mapping = class_to_idx_mapping
        self.features, self.targets = self.get_features_labels(feature_dict)
        self.features_dim = self.features[0].shape[0]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], int(self.targets[idx])

    def get_features_labels(self, feature_dict):
        features, targets = [], []
        for label in self.labels:
            feats = feature_dict[label]
            targets.extend([self.class_to_idx_mapping[label]] * len(feats))
            features.append(feats)
        return torch.cat(features), torch.Tensor(targets)

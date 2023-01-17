import logging
from collections import defaultdict

import numpy as np
import torch
import wandb
from datasets.zoc_loader import IsolatedClasses
from ood_detection.classification_utils import zeroshot_classifier
from ood_detection.config import Config
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from zoc.utils import get_ablation_splits, get_split_specific_targets, fill_auc_lists, fill_f_acc_lists, \
    get_result_mean_dict

_logger = logging.getLogger(__name__)


@torch.no_grad()
def get_feature_weight_dict(isolated_classes, clip_model, device):
    weights_dict = {}
    for cls in isolated_classes.labels:
        loader = isolated_classes[cls]
        image_feature_list = []
        for images in tqdm(loader):
            images = images.to(device)
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            image_feature_list.append(image_features)
        weights_dict[cls] = torch.cat(image_feature_list).half()

    return weights_dict


def get_zeroshot_weight_dict(isolated_classes, clip_model):
    weights_dict = {}
    weights = zeroshot_classifier(isolated_classes.labels, isolated_classes.templates, clip_model)

    for classname, weight in zip(isolated_classes.labels, weights):
        weights_dict[classname] = weight

    return weights_dict


def sorted_zeroshot_weights(weights, split):
    sorted_weights = []
    for classname in split:
        sorted_weights.append(weights[classname])
    return torch.stack(sorted_weights)


@torch.no_grad()
def baseline_detector(clip_model,
                      device,
                      isolated_classes: IsolatedClasses = None,
                      id_classes=6,
                      ood_classes=4,
                      runs=1, ):
    feature_weight_dict = get_feature_weight_dict(isolated_classes, clip_model, device)
    classes_weight_dict = get_zeroshot_weight_dict(isolated_classes, clip_model)

    ablation_splits = get_ablation_splits(isolated_classes.labels, n=runs, id_classes=id_classes,
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
                zeroshot_probs = (temperature * image_features_for_label.to(torch.float32) @ zeroshot_weights.T.to(
                    torch.float32)).softmax(dim=-1).squeeze()

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


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


def train_id_classifier(train_set, eval_set):
    classifier = LinearClassifier(train_set.features_dim, len(train_set.labels))

    train_loader = DataLoader(train_set,
                              batch_size=512,
                              shuffle=True)
    eval_loader = DataLoader(eval_set,
                             batch_size=512,
                             shuffle=True)

    epochs = 10
    learning_rate = 0.001
    optimizer = AdamW(params=classifier.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    best_eval_loss = np.inf
    for epoch in tqdm(range(1, epochs + 1)):

        epoch_results = {}
        epoch_loss = 0.
        # train
        for image_features, targets in tqdm(train_loader):
            optimizer.zero_grad()

            preds = classifier(image_features)
            output = criterion(preds, targets)
            output.backward()
            epoch_loss += output

            optimizer.step()

        epoch_results["epoch"] = epoch
        epoch_results["train loss"] = epoch_loss
        epoch_results["train loss per image"] = epoch_loss / len(train_loader)

        # eval

        epoch_val_loss = 0.
        for eval_features, eval_targets in tqdm(eval_loader):
            with torch.no_grad():
                eval_preds = classifier(eval_features)
                eval_loss = criterion(eval_preds, eval_targets).detach().item()

            epoch_val_loss += eval_loss

            if epoch_val_loss < best_eval_loss:
                best_eval_loss = epoch_val_loss
                best_classifier = LinearClassifier.weights

            _, indices = torch.topk(torch.softmax(eval_preds, k=1, dim=-1))
            accuracy = accuracy_score(eval_targets, indices)
            _logger.info(f"Epoch {epoch} Eval Acc: {accuracy}")

            epoch_results["val loss"] = epoch_val_loss
            epoch_results["train loss per image"] = epoch_val_loss / len(eval_loader)

            wandb.log(epoch_results)
    return best_classifier


def linear_layer_detector(dataset, clip_model, clip_transform, id_classes, ood_classes, runs):
    isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                               split='train',
                                               transform=clip_transform),
                                       batch_size=512)
    feature_weight_dict_train = get_feature_weight_dict(isolated_classes, clip_model, Config.DEVICE)

    isolated_classes = IsolatedClasses(dataset(Config.DATAPATH,
                                               split='val',
                                               transform=clip_transform),
                                       batch_size=512)
    feature_weight_dict_val = get_feature_weight_dict(isolated_classes, clip_model, Config.DEVICE)
    ablation_splits = get_ablation_splits(isolated_classes.labels, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)
    for ablation_split in ablation_splits:
        # train classifier to classify id set
        train_set = FeatureSet(feature_weight_dict_train, ablation_split[:id_classes])
        val_set = FeatureSet(feature_weight_dict_val, ablation_split[:id_classes])

        run = wandb.init(project="thesis-linear clip",
                         entity="wandbefab",
                         name=dataset.get_name(),
                         tags=[
                             'linear probe',
                             'oodd',
                         ])
        classifier = train_id_classifier(train_set, val_set)
        run.finish()
        print("DONE")
        # eval for ood detection
        return "FINISH"

class FeatureSet(Dataset):
    def __init__(self, feature_dict, labels, dataset):
        self.labels = labels
        self.dataset = dataset
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
            targets.extend([self.dataset.class_to_idx] * len(feats))
            features.append(feats)
        return torch.cat(features), torch.Tensor(targets)

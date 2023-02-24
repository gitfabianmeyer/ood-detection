import logging
import random
from abc import ABC, abstractmethod

import clip
import numpy as np
import torch

from sklearn.metrics.pairwise import rbf_kernel
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from datasets import corruptions
from datasets.classnames import imagenet_templates, base_template
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config
from ood_detection.classification_utils import zeroshot_classifier, classify
from metrics.distances_utils import id_ood_printer, mean_std_printer, \
    distance_name_printer, accuracy_printer
from zeroshot.classification import get_cosine_similarity_matrix_for_normed_features

from zeroshot.utils import FeatureDict
from zoc.baseline import sorted_zeroshot_weights
from zoc.utils import get_image_features_for_isolated_class_loader

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class Distancer:
    def __init__(self, isolated_classes, clip_model, splits=5, id_split=.4):
        self.splits = splits
        self.id_split = id_split
        self.dataloaders = isolated_classes
        self.classes = isolated_classes.classes
        self.clip_model = clip_model.eval()
        self.device = Config.DEVICE
        self.feature_dict = {}
        self.get_feature_dict()
        self.targets = self.get_dataset_targets()

    def get_feature_dict(self, max_len=20000):
        _logger.info("Start creating image features...")
        max_per_class = max_len // len(self.classes)
        for cls in tqdm(self.classes):
            self.feature_dict[cls] = get_image_features_for_isolated_class_loader(self.dataloaders[cls],
                                                                                  self.clip_model, max_per_class)

    def get_mmd(self):
        distance_name_printer("MMD")
        mmd = MaximumMeanDiscrepancy(self.feature_dict)
        return mmd.get_distance_for_n_splits(splits=self.splits)

    def get_clp(self):
        distance_name_printer("CLP")
        clp = ConfusionLogProbability(self.feature_dict, self.clip_model)
        return clp.get_distance_for_n_splits(self.splits)

    def get_zeroshot_accuracy(self):
        distance_name_printer("Zero Shot Accuracy")
        zsa = ZeroShotAccuracy(self.feature_dict,
                               self.clip_model,
                               self.targets)
        return zsa.get_distance()

    def get_all_distances(self):
        mmd_mean, mmd_std = self.get_mmd()
        mean_std_printer(mmd_mean, mmd_std, self.splits)

        clp_mean, clp_std = self.get_clp()
        mean_std_printer(clp_mean, clp_std, self.splits)

        zsa_dict = self.get_zeroshot_accuracy()
        accuracy_printer(zsa_dict["zsa"])
        res_dict = {"mmd_mean": mmd_mean,
                    "mmd_std": mmd_std,
                    "clp_mean": clp_mean,
                    "clp_std": clp_std}

        return {**res_dict, **zsa_dict}

    def get_dataset_targets(self):
        targets = []
        for dataloader in self.dataloaders.values():
            targets.append(dataloader.dataset.targets)
        return torch.cat(targets)


class Distance(ABC):
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict
        self.classes = list(self.feature_dict.keys())

    def get_distance_for_n_splits(self, splits=5):
        distances = [self.get_distance() for _ in range(splits)]
        return np.mean(distances), np.std(distances)

    def get_distribution_features(self, classes):
        return torch.cat([self.feature_dict[cla] for cla in classes])

    def get_id_ood_split(self, in_distri_percentage=.4):
        random.shuffle(self.classes)
        if len(self.classes) == 2:
            return self.classes[:1], self.classes[1:]
        id_split = int(len(self.classes) * in_distri_percentage)
        return self.classes[:id_split], self.classes[id_split:]

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_distance(self):
        pass


class ZeroShotAccuracy(Distance):
    def __init__(self, feature_dict, clip_model, dataset_targets):
        super(ZeroShotAccuracy, self).__init__(feature_dict)
        self.clip_model = clip_model
        self.labels = zeroshot_classifier(self.classes, imagenet_templates, self.clip_model)
        self.dataset_targets = dataset_targets

    @property
    def name(self):
        return 'Zero Shot Accuracy'

    def get_distance(self):
        # do for the whole set
        top1 = classify(features=torch.cat(list(self.feature_dict.values())),
                        zeroshot_weights=self.labels,
                        targets=self.dataset_targets)

        class_lengths = [len(feat_class) for feat_class in self.feature_dict.values()]
        len_dataset = sum(class_lengths)
        len_max_class = max(class_lengths)
        results = {"zsa": top1,
                   "zsa_baseline": len_max_class / len_dataset}
        return results


class MaximumMeanDiscrepancy(Distance):
    def __init__(self, feature_dict):
        super(MaximumMeanDiscrepancy, self).__init__(feature_dict)
        self.kernel_size = self.get_kernel_size()

    def get_distance(self):
        # for near OOD
        id_classes, ood_classes = self.get_id_ood_split()
        id_ood_printer(id_classes, ood_classes)
        id_features = self.get_distribution_features(id_classes).cpu().numpy()
        ood_features = self.get_distribution_features(ood_classes).cpu().numpy()
        return self.get_mmd(x_matrix=id_features,
                            y_matrix=ood_features)

    @property
    def name(self):
        return "Maximum Mean Discrepancy"

    def get_mmd(self, x_matrix, y_matrix):
        batch_size = x_matrix.shape[0]
        beta = (1. / (batch_size * (batch_size - 1)))

        gamma = (2. / (batch_size * batch_size))

        XX = rbf_kernel(x_matrix, x_matrix, self.kernel_size)
        YY = rbf_kernel(y_matrix, y_matrix, self.kernel_size)
        XY = rbf_kernel(x_matrix, y_matrix, self.kernel_size)

        return beta * (XX.sum() + YY.sum()) - gamma * XY.sum()

    def get_kernel_size(self):
        X = torch.cat(list(self.feature_dict.values()))
        return torch.mean(torch.cdist(X, X)).cpu().numpy()


class ConfusionLogProbability(Distance):

    @property
    def name(self):
        return "Confusion Log Probability"

    def __init__(self, feature_dict, clip_model):
        super(ConfusionLogProbability, self).__init__(feature_dict)
        self.clip_model = clip_model
        self.class_features_dict = self.get_class_features_dict()

    def get_distance(self, temperature=0.01):
        id_classes, ood_classes = self.get_id_ood_split()

        # get labels AFTER shuffling self.classes in get_id_ood_split()
        # classes are shuffled by get_id_ood_split()!
        labels = sorted_zeroshot_weights(self.class_features_dict, self.classes)

        ood_features = torch.cat([self.feature_dict[ood_class] for ood_class in ood_classes])

        logits = get_cosine_similarity_matrix_for_normed_features(ood_features, labels, 0.01)

        # debug_scores(logits, "Logits")
        softmax_scores = F.softmax(logits, dim=1)
        # debug_scores(softmax_scores, "Softmaxis")

        id_scores = softmax_scores[:, :len(id_classes)]  # use only id labels proba
        confusion_log_proba = torch.log(id_scores.sum(dim=1).mean())
        return confusion_log_proba.cpu().numpy()

    def get_class_features_dict(self):
        features = zeroshot_classifier(self.classes, imagenet_templates, self.clip_model)
        return {label: value for (label, value) in zip(self.classes, features)}


def get_far_clp(id_dict: FeatureDict, ood_dict: FeatureDict, clip_model, temperature):
    id_classes = id_dict.classes
    ood_classes = ood_dict.classes
    classes = id_classes + ood_classes
    zsw = zeroshot_classifier(classes, base_template, clip_model)
    features = ood_dict.get_features()
    logits = get_cosine_similarity_matrix_for_normed_features(features, zsw, temperature)

    softmax_scores = F.softmax(logits, dim=1)
    id_scores = softmax_scores[:, :len(id_classes)]  # use only id labels proba
    confusion_log_proba = torch.log(id_scores.sum(dim=1).mean())

    return confusion_log_proba.cpu().numpy()


def get_mmd_rbf_kernel(id_features, ood_features):
    import math
    X = torch.cat((id_features, ood_features)).to(torch.float32).cpu()
    kernel_size = torch.mean(torch.cdist(X, X)).cpu().numpy()
    float_kernel = float(kernel_size)
    if math.isnan(float_kernel):
        raise ValueError("Kernel is NAN")
    return float_kernel


def get_far_mmd(id_dict: FeatureDict, ood_dict: FeatureDict):
    x_matrix = id_dict.get_features()
    y_matrix = ood_dict.get_features()
    try:
        kernel_size = get_mmd_rbf_kernel(x_matrix, y_matrix)
    except ValueError as e:
        _logger.error(F"Kernel is NAN")
        raise e

    batch_size = x_matrix.shape[0]
    beta = (1. / (batch_size * (batch_size - 1)))

    gamma = (2. / (batch_size * batch_size))

    _logger.warning(f"kernel: {kernel_size}\nbeta:{beta}\ngamma:{gamma}") # TODO
    x_matrix = x_matrix.detach().cpu().numpy()
    y_matrix = y_matrix.detach().cpu().numpy()
    XX = rbf_kernel(x_matrix, x_matrix, kernel_size)
    YY = rbf_kernel(y_matrix, y_matrix, kernel_size)
    XY = rbf_kernel(x_matrix, y_matrix, kernel_size)

    return beta * (XX.sum() + YY.sum()) - gamma * XY.sum()


class WassersteinDistance(Distance):

    @property
    def name(self):
        return "Wasserstein Distance"

    def get_distance(self):
        pass


def get_distances_for_dataset(dataset, clip_model, splits=10, id_split=.4, corruption=None, severity=None,
                              lsun=False):
    loaders = IsolatedClasses(dataset, batch_size=512, lsun=lsun)
    distancer = Distancer(isolated_classes=loaders,
                          clip_model=clip_model,
                          splits=splits,
                          id_split=id_split)
    logging_dict = distancer.get_all_distances()
    logging_dict['id_split_size'] = id_split
    logging_dict["splits"] = splits
    if corruption and severity:
        logging_dict["severity"] = severity
    return logging_dict


def get_corruption_metrics(dataset, clip_model, clip_transform, dataset_name, lsun=False, split='val'):
    data_path = Config.DATAPATH

    corruption_dict = corruptions.Corruptions
    for name, corri in corruption_dict.items():
        for i in range(1, 6):
            print(f"Corruption {name}, severity: {i}")
            corruption = corri(severity=i)
            transform_list = clip_transform.transforms[:-2]
            transform_list.append(corruption)
            transform_list.extend(clip_transform.transforms[-2:])
            transform = Compose(transform_list)
            dset = dataset(data_path, transform, split)
            run = get_distances_for_dataset(dset, clip_model, dataset_name, lsun=lsun, corruption=name, severity=i)
        run.finish()


def run_full_distances(name, dataset, lsun=False, split='val'):
    data_path = Config.DATAPATH
    clip_model, transform_clip = clip.load(Config.VISION_MODEL)

    get_corruption_metrics(dataset=dataset,
                           clip_model=clip_model,
                           clip_transform=transform_clip,
                           dataset_name=name,
                           lsun=lsun,
                           split=split)

    dataset = dataset(data_path, transform_clip, split)
    get_distances_for_dataset(dataset, clip_model, name, lsun=lsun)

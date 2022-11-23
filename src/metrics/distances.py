import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from datasets.classnames import imagenet_templates
from datasets.zoc_loader import single_isolated_class_loader
from ood_detection.config import Config
from ood_detection.classification_utils import zeroshot_classifier
from sklearn.metrics.pairwise import rbf_kernel
import torch.nn.functional as F
from tqdm import tqdm

from metrics.distances_utils import id_ood_printer, \
    shape_printer, dataset_name_printer, mean_std_printer, distance_name_printer


class Distancer:
    def __init__(self, dataloaders, clip_model, splits=5):
        self.splits = splits
        self.dataloaders = dataloaders
        self.classes = list(self.dataloaders.keys())
        self.clip_model = clip_model.eval()
        self.device = Config.DEVICE
        self.feature_dict = {}
        self.get_feature_dict()

    def get_image_batch_features(self, loader, stop_at=None):
        with torch.no_grad():
            features = []
            for images in loader:
                images = images.to(self.device)
                batch_features = self.clip_model.encode_image(images)
                batch_features /= batch_features.norm(dim=1, keepdim=True)
                features.append(batch_features)
                if len(features) >= stop_at:
                    print(f"Reached max {stop_at} for class {loader.name}")
                    break

            return torch.cat(features)

    def get_feature_dict(self, max_len=20000):
        print("Start creating image features...")
        max_per_class = max_len // len(self.classes)
        for cls in tqdm(self.classes):
            self.feature_dict[cls] = self.get_image_batch_features(self.dataloaders[cls], max_per_class)

    def get_mmd(self):
        distance_name_printer("MMD")
        mmd = MaximumMeanDiscrepancy(self.feature_dict)
        return mmd.get_distance_for_n_splits(splits=self.splits)

    def get_clp(self):
        distance_name_printer("CLP")
        clp = ConfusionLogProbability(self.feature_dict, self.clip_model)
        return clp.get_distance_for_n_splits(self.splits)

    def get_all_distances(self):
        mmd_mean, mmd_std = self.get_mmd()
        mean_std_printer(mmd_mean, mmd_std, self.splits)

        clp_mean, clp_std = self.get_clp()
        mean_std_printer(clp_mean, clp_std, self.splits)


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
        id_split = int(len(self.classes) * in_distri_percentage)
        return self.classes[:id_split], self.classes[id_split:]

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_distance(self):
        pass


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
        self.labels = zeroshot_classifier(self.classes, imagenet_templates, self.clip_model)

    def get_distance(self):
        id_classes, ood_classes = self.get_id_ood_split()
        id_ood_printer(id_classes, ood_classes)
        ood_features = torch.cat([self.feature_dict[ood_class] for ood_class in ood_classes])
        # shape_printer("ood features", ood_features)
        # shape_printer("labels features", self.labels)

        logits = ood_features.half() @ self.labels.t().half()
        # shape_printer("logits", logits)
        softmax_scores = F.softmax(logits, dim=1)
        # shape_printer("Softmax Scores", softmax_scores)
        id_scores = softmax_scores[:, :len(id_classes)]  # use only id labels proba
        confusion_log_proba = torch.log(id_scores.sum(dim=1).mean())
        return confusion_log_proba.cpu().numpy()


def get_distances_for_dataset(dataset, clip_model, name):
    dataset_name_printer(name)
    loaders = single_isolated_class_loader(dataset, batch_size=512)
    distancer = Distancer(dataloaders=loaders,
                          clip_model=clip_model,
                          splits=10)
    distancer.get_all_distances()
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from datasets.classnames import imagenet_templates
from ood_detection.config import Config
from ood_detection.ood_utils import zeroshot_classifier
from sklearn.metrics.pairwise import rbf_kernel
import torch.nn.functional as F
from tqdm import tqdm

from metrics.distances_utils import id_ood_printer, shape_printer


class Distance(ABC):
    def __init__(self, dataloaders, clip_model):
        self.dataloaders = dataloaders
        self.classes = list(self.dataloaders.keys())
        self.clip_model = clip_model.eval()
        self.device = Config.DEVICE
        self.feature_dict = {}
        self.get_feature_dict()

    def get_distance_for_n_splits(self, splits=5):
        distances = [self.get_distance() for _ in range(splits)]
        return np.mean(distances), np.std(distances)

    def get_distribution_features(self, classes):
        return torch.cat([self.feature_dict[cla] for cla in classes])

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_distance(self):
        pass

    def get_image_batch_features(self, loader):
        with torch.no_grad():
            features = []
            for images in loader:
                images = images.to(self.device)
                batch_features = self.clip_model.encode_image(images)
                batch_features /= batch_features.norm(dim=1, keepdim=True)
                features.append(batch_features)

            return torch.cat(features)

    def get_id_ood_split(self, in_distri_percentage=.4):
        random.shuffle(self.classes)
        id_split = int(len(self.classes) * in_distri_percentage)
        return self.classes[:id_split], self.classes[id_split:]

    def get_feature_dict(self):
        print("Start obtaining features)")
        for cls in tqdm(self.classes):
            self.feature_dict[cls] = self.get_image_batch_features(self.dataloaders[cls])


class MaximumMeanDiscrepancy(Distance):
    def __init__(self, dataloaders, clip_model, ):
        super(MaximumMeanDiscrepancy, self).__init__(dataloaders,
                                                     clip_model)
        self.kernel_size = self.get_kernel_size()

    def get_distance(self):
        # for near OOD
        id_classes, ood_classes = self.get_id_ood_split()
        print(f"id Classes: {id_classes}\n\n OOD classes: {ood_classes}")
        id_features = self.get_distribution_features(id_classes).cpu().numpy()
        ood_features = self.get_distribution_features(ood_classes).cpu().numpy()
        return self.get_mmd(x_matrix=id_features,
                            y_matrix=ood_features)

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
        print(f"Start calculating RBF kernel size")
        X = torch.cat(list(self.feature_dict.values()))
        return torch.mean(torch.cdist(X, X)).cpu().numpy()


class ConfusionLogProbability(Distance):
    @property
    def name(self):
        return "Confusion Log Probability"

    def __init__(self, dataloaders, clip_model):
        super(ConfusionLogProbability, self).__init__(dataloaders, clip_model)
        self.labels = zeroshot_classifier(self.classes, imagenet_templates, self.clip_model)

    def get_distance(self):
        id_classes, ood_classes = self.get_id_ood_split()
        id_ood_printer(id_classes, ood_classes)
        ood_features = torch.cat([self.feature_dict[ood_class] for ood_class in ood_classes])
        shape_printer("ood features", ood_features)
        shape_printer("labels features", self.labels)

        logits = ood_features.half() @ self.labels.half()
        shape_printer("logits", logits)
        softmax_scores = F.softmax(logits, dim=1)
        shape_printer("Softmax Scores", softmax_scores)
        id_scores = softmax_scores[:, len(id_classes)]  # use only id labels proba
        confusion_log_proba = torch.log(id_scores.sum(dim=1).mean())
        return confusion_log_proba

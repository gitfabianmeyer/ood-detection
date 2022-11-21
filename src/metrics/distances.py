import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from ood_detection.config import Config
from sklearn import metrics
from tqdm import tqdm


class Distance(ABC):
    def __init__(self, dataloaders, classes, clip_model):
        self.dataloaders = dataloaders
        self.classes = classes
        self.clip_model = clip_model.eval()
        self.device = Config.DEVICE
        self.feature_dict = {}
        self.get_feature_dict()

    def get_distance_for_n_splits(self, splits=5):
        return np.mean([self.get_distance() for _ in range(splits)])

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
    def __init__(self, dataloaders, classes, clip_model, ):
        super(MaximumMeanDiscrepancy, self).__init__(dataloaders,
                                                     classes,
                                                     clip_model)
        self.kernel_size = self.get_kernel_size()

    def get_distance(self):
        # for near OOD
        id_classes, ood_classes = self.get_id_ood_split()
        id_features = self.get_distribution_features(id_classes)
        ood_features = self.get_distribution_features(ood_classes)
        return self.get_mmd(X=id_features,
                            Y=ood_features)

    def name(self):
        return "Maximum Mean Discrepancy"

    def get_mmd(self, X, Y, kernel_size):
        assert X.shape == Y.shape, f"X1: {X.shape} does not match X2: {Y.shape}"
        batch_size = X.shape[0]
        beta = (1. / (batch_size * (batch_size - 1)))
        gamma = (2. / (batch_size * batch_size))

        XX = metrics.pairwaise.rbf_kernel(X,X, kernel_size)
        YY = metrics.pairwise.rbf_kernel(Y, Y, kernel_size)
        XY = metrics.pairwise.rbf_kernel(X,Y, kernel_size)
        return beta * (XX.mean() + YY.mean()) - gamma * np.mean(XY)

    def get_kernel_size(self):
        X = torch.cat(list(self.feature_dict.values()))
        return torch.mean(torch.cdist(X, X))

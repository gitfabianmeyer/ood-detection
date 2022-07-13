import os
from collections import defaultdict
from typing import Dict

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ood_detection.config import Config


def plot_pca_analysis(run_directory, n_components=2):

    features = {}

    # load all features
    for _, _, file in os.walk(run_directory):
        feature_name = file.split("_")[0]
        features[feature_name] = torch.load(file)

    # do pca on whole set of features
    full = torch.stack(features.values())
    pca = PCA(n_components=n_components)
    pca.fit(full)
    plt.figure(figsize=(16, 16))

    # transform each feature tensor with pca and plot
    for k, v in features.items():
        pca_data = pca.transform(v)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=.5, label=k)
    plt.legend()
    plt.savefig(os.path.join(Config.PLOTS, run_directory))

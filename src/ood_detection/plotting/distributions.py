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
    for root, cwd, files in os.walk(run_directory):

        collected_files = []
        for file in files:
            if file.split("_")[2] == 't.pt':
                print(file)
                # ignore targets
                continue
            print(f"INFILE: {file}")
            feature_name = file.split("_")[0]
            if torch.cuda.is_available():
                features[feature_name] = torch.load(os.path.join(root, file))
            else:
                features[feature_name] = torch.load(os.path.join(root, file), map_location='cpu')
            collected_files.append(feature_name)

    # do pca on whole set of features
    full = torch.cat(list(features.values()), dim=0)
    print(f'Full tensor shape: {full.shape}')
    pca = PCA(n_components=n_components)
    pca.fit(full)
    plt.figure(figsize=(16, 16))

    # transform each feature tensor with pca and plot
    for k, v in features.items():
        pca_data = pca.transform(v)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=.5, label=k)
    plt.legend()

    file_ending = run_directory.split("/")[-1]
    plt.savefig(os.path.join(Config.PLOTS, file_ending))

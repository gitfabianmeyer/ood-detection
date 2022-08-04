import os

import seaborn
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

id_set = "pets"
base_path = "link"
dataset_features_path = "link"
labels_id_path = "link"
labels_ood_path = "link"


def load_ood_features(base_path):
    files = {}
    for root, _, file in os.walk(base_path):
        if file.endswith("ood.pt"):
            f = torch.load(os.path.join(root, file))
            files[file] = f
    return files


dataset_features = torch.load(dataset_features_path)
labels_id = torch.load(labels_id_path)
labels_ood = torch.load(labels_ood_path) # TODO write oodlabel features generator
ood_features = load_ood_features(base_path)
ood_features_flat = torch.cat([tensor for tensor in ood_features.values()])

full_set = torch.cat(dataset_features, labels_id, labels_ood, ood_features_flat)
print(f"Shape of full data set: {full_set.shape}")
pca = PCA(n_components=2)
pca.fit(full_set)

pca_dataset = pca.transform(dataset_features)
pca_labels_id = pca.transform(labels_id)

plt.figure(figsize=(8, 8))
plt.scatter(pca_dataset[:, 0], pca_dataset[:, 1], color='blue', alpha=.5, label=id_set)
plt.scatter(pca_labels_id[:, 0], pca_labels_id[:, 1], color='green', alpha=.5, label="ID labels")

for key, value in ood_features.items():
    plt.scatter(value[:, 0], value[:, 1], alpha=.5, label=key)


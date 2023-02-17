import numpy as np
import torch
from ood_detection.config import Config


@torch.no_grad()
def get_image_features_for_isolated_class_loader(loader, clip_model, stop_at=np.inf):
    features = []
    for images in loader:
        features.append(get_normalized_image_features(clip_model, images))
        if len(features) >= stop_at:
            print(f"Reached max {stop_at} for class {loader.name}")
            break

    return torch.cat(features)


@torch.no_grad()
def get_image_features_and_targets(loader, clip_model, stop_at=np.inf):
    features, targets = [], []
    for images, targets in loader:

        targets = targets.to(Config.DEVICE)
        targets.append(targets)

        features.append(get_normalized_image_features(clip_model, images))

        if len(features) >= stop_at:
            print(f"Reached max {stop_at} for class {loader.name}")
            break

    return torch.cat(features), torch.cat(targets)


@torch.no_grad()
def get_normalized_image_features(clip_model, images):
    images = images.to(Config.DEVICE)
    image_features = clip_model.encode_image(images)
    image_features /= image_features.norm(dim=1, keepdim=True)
    return images

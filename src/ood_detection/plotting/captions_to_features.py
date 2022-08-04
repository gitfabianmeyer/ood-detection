import clip
import torch
from ood_detection.classnames import imagenet_templates
from ood_detection.config import Config
from ood_detection.ood_utils import zeroshot_classifier


def captions_to_features(path_to_texfile):

    rel_path = "/".join(path_to_texfile.split("/")[:-1])
    with open(path_to_texfile, 'r', encoding='utf-8') as f:
        captions = [caption.rstrip('\n').split(" ") for caption in f]
    print(f"Number of captions: {len(captions)}")
    # clean everything
    cleaned_labels = set()
    for caption in captions:
        for label in caption:
            cleaned_labels.add(label)

    # get features
    clip_model = clip.load(Config.VISION_MODEL)
    features = zeroshot_classifier(list(cleaned_labels), imagenet_templates, clip_model)
    torch.save(features, rel_path)
    return features

from typing import List

import PIL
import clip
import torch
from nltk.corpus import stopwords
from ood_detection.classnames import imagenet_templates
from ood_detection.config import Config

from ood_detection.models.dummy_zoc import CaptionGenerator
from ood_detection.ood_classification import get_ood_targets, classify

from src.ood_detection.ood_utils import get_individual_ood_weights

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
stopwords = set(stopwords.words('english'))


def remove_stopwords(caption, stop_words=stopwords):
    return [word for word in caption.split(" ") if word not in stop_words]


def run_batch_ood(image: PIL.Image,
                  target,
                  id_labels: List[str],
                  caption_generator: CaptionGenerator,
                  clip_model,
                  clip_preprocess,
                  zero_shot_weights,
                  stop_words=None):
    caption = caption_generator.generate_caption(image)
    if stop_words:
        ood_label = remove_stopwords(caption, stop_words)

    else:
        ood_label = [word for word in caption.split(" ")]

    print(f"Label: {ood_label}")


    # append individual labels to the zeroshot weights
    zero_shot_weights.append(get_individual_ood_weights(ood_label,
                                                        clip_model,
                                                        templates=imagenet_templates))
    zero_shot_weights = torch.stack(zero_shot_weights, dim=1).to(device)

    clip_model.eval()
    image = image.to(device)
    target = target.to(device)
    image_features = clip_model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    #classify(ood_features, zero_shot_weights, ood_labels, name)
    # rewrite for 1 image, see if it works
    return ood_label

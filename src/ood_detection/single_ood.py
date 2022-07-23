import os

import PIL
import clip
import torch
from PIL import Image
from nltk.corpus import stopwords
from ood_detection.classnames import imagenet_templates, stanfordcars_classes
from ood_detection.config import Config

from ood_detection.models.dummy_zoc import CaptionGenerator
from transformers import GPT2Tokenizer

from src.ood_detection.ood_utils import get_individual_ood_weights, zeroshot_classifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
stopwords = set(stopwords.words('english'))


def remove_stopwords(caption, stop_words=stopwords):
    return [word for word in caption.split(" ") if word not in stop_words]


def run_batch_ood(image: PIL.Image,
                  target,
                  caption_generator: CaptionGenerator,
                  clip_model,
                  preprocess,
                  zero_shot_weights,
                  stop_words=None):
    # generate the pseudo labels aka the caption
    image = preprocess(image).unsqueeze(0).to(device)
    caption = caption_generator.generate_caption(image)
    if stop_words:
        ood_label = remove_stopwords(caption, stop_words)

    else:
        ood_label = [word for word in caption.split(" ")]

    print(f"Label: {ood_label}")

    # append individual labels to the zeroshot weights
    ind_class_embeddings = get_individual_ood_weights(ood_label,
                                                      clip_model,
                                                      templates=imagenet_templates)

    zeroshot_weights = torch.cat([zero_shot_weights, ind_class_embeddings], dim=0).to(device)

    clip_model.eval()
    image = image.to(device)
    # target = target.to(device)
    image_features = clip_model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    logits = 100. * image_features @ zeroshot_weights

    # acc
    pred = logits.topk(max((1,)), 1, True, True)[1].t()
    print(f"Predicted: {pred}")
    print(f"True: {target}")

    # classify(ood_features, zeroshot_weights, ood_labels, name)
    # rewrite for 1 image, see if it works
    return pred


def main():
    # initialize everything
    model_path = os.path.join(Config.MODELS, 'generator_weights.pt')

    print(f"Loading CLIP with Vision Modul: {Config.VISION_MODEL}...")
    clip_model, preprocess = clip.load(Config.VISION_MODEL)
    clip_model.eval()
    print("Loading GPT2 tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Initializing CaptionGenerator")
    caption_generator = CaptionGenerator(model_path=model_path,
                                         clip_model=clip_model,
                                         tokenizer=tokenizer,
                                         prefix_length=10)

    templates = imagenet_templates
    classnames = stanfordcars_classes

    # get the label features
    zeroshot_weights = zeroshot_classifier(classnames, templates, clip_model)

    # get an image and according label
    image = 'stanford_cars/cars_train/00003.jpg'
    image_path = os.path.join(Config.DATAPATH, image)
    im = Image.open(image_path)

    run_batch_ood(im,
                  target=80,
                  caption_generator=caption_generator,
                  clip_model=clip_model,
                  preprocess=preprocess,
                  zero_shot_weights=zeroshot_weights,
                  stop_words=stopwords)


if __name__ == '__main__':
    print("Starting run")
    main()

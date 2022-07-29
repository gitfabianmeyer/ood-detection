import os
import time
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import GPT2Tokenizer

import PIL
import clip
import torch
import torchvision.datasets
from PIL import Image
from nltk.corpus import stopwords
from ood_detection.classnames import imagenet_templates, stanfordcars_classes, oxfordpets_classes
from ood_detection.config import Config

from ood_detection.models.dummy_zoc import CaptionGenerator
from tqdm import tqdm

from ood_detection.ood_utils import get_individual_ood_weights, zeroshot_classifier, \
    accuracy, save_features_labels_captions, \
    load_features_labels_captions, get_full_logits, generate_caption_with_generator

device = Config.DEVICE
print(f"Using {device}")
stopwords = set(stopwords.words('english'))
batch_size = 256
cut_off_labels = 5
load_zeroshot = True
load_features = True


def remove_stopwords(caption, stop_words=stopwords):
    return [word for word in caption.split(" ") if word not in stop_words]


def run_batch_ood(image: PIL.Image,
                  target,
                  caption_generator: CaptionGenerator,
                  clip_model,
                  preprocess,
                  class_weights,
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

    zeroshot_weights = torch.cat([class_weights, ind_class_embeddings], dim=1).to(device)

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


def main(generate_caption=True):
    print("Starting run")
    # initialize everything

    in_distri_set = "pets"
    run = in_distri_set
    # run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    curr_datapath = os.path.join(Config.FEATURES, run)
    os.makedirs(curr_datapath, exist_ok=True)

    model_path = os.path.join(Config.MODELS, 'generator_weights.pt')
    zeroshot_path = os.path.join(curr_datapath, f'{in_distri_set}_zeroshot.pt')
    features_path = os.path.join(curr_datapath, f'{in_distri_set}_features.pt')
    targets_path = os.path.join(curr_datapath, f'{in_distri_set}_targets.pt')
    ind_labels_path = os.path.join(curr_datapath, f'{in_distri_set}_ind_labels.pt')
    captions_path = os.path.join(curr_datapath, f'{in_distri_set}_captions.txt')
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

    dataset = torchvision.datasets.OxfordIIITPet(Config.DATAPATH,
                                                 transform=preprocess)
    templates = imagenet_templates
    classnames = dataset.classes

    # get the label features
    if not load_zeroshot:
        zeroshot_weights = zeroshot_classifier(classnames, templates, clip_model)
        torch.save(zeroshot_weights, zeroshot_path)
        print(f"Saved zsw at {zeroshot_path}")
    else:
        if torch.cuda.is_available():
            zeroshot_weights = torch.load(zeroshot_path)
        else:
            zeroshot_weights = torch.load(zeroshot_path, map_location=torch.device('cpu'))
        print("Loaded zeroshot weights")

    print(f"Classifying {len(dataset._images)} images")
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=8)

    features = []
    labels = []
    full_captions = []
    if not load_features:
        with torch.no_grad():
            for images, targets in tqdm(dataloader):
                images = images.to(device)
                targets = targets.to(device)
                images_features = clip_model.encode_image(images).to(device, dtype=torch.float32)

                if generate_caption:
                    captions = generate_caption_with_generator(caption_generator, images_features, classnames, cut_off_labels)
                    full_captions.extend(captions)
                images_features /= images_features.norm(dim=-1, keepdim=True)
                features.append(images_features)
                labels.append(targets)

            features = torch.cat(features)
            labels = torch.cat(labels)

        save_features_labels_captions(features, features_path, labels, targets_path)

    else:
        features, labels, full_captions = load_features_labels_captions(features_path, targets_path, captions_path)

    full_logits = get_full_logits(features, zeroshot_weights, full_captions, clip_model, imagenet_templates)

    print(
        f"Shape should be: {len(dataset.classes) + cut_off_labels} x {len(dataset._images)} and is: {full_logits.shape}")
    # no adapation needed, as new labels are always wrong

    # TODO or fix it here to use labels each time. OR append 5 labels
    top1, top5, n = 0., 0., 0.
    acc1, acc5 = accuracy(full_logits, labels, top_k=(1, 5))
    top1 += acc1
    top5 += acc5

    n += features.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"\nClip Top1 Acc: {top1:.2f} with zeroshot")
    print(f"\nClip Top5 Acc: {top5:.2f} with zeroshot")


if __name__ == '__main__':
    main()

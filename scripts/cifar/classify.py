import os

from src.ood_detection.ood_utils import ood_accuracy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
from collections import defaultdict

import clip
import torch
import torchvision.datasets
from ood_detection.classnames import imagenet_templates
from ood_detection.config import Config

# init stuff
from ood_detection.models.dummy_zoc import CaptionGenerator
from ood_detection.ood_utils import zeroshot_classifier, \
    clean_caption, save_features_labels_captions, \
    load_features_labels_captions, get_full_logits, \
    generate_caption_with_generator, accuracy

from tqdm import tqdm
from transformers import GPT2Tokenizer


def main():
    print("Starting run")
    load_features = False
    load_zeroshot = False
    batch_size = 8
    cut_off_labels = 5
    generate_captions = False

    ood_set = "cifar"
    run = ood_set
    # run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    curr_datapath = os.path.join(Config.FEATURES, run)
    os.makedirs(curr_datapath, exist_ok=True)

    model_path = os.path.join(Config.MODELS, 'generator_weights.pt')
    zeroshot_path = os.path.join(curr_datapath, f'{ood_set}_zeroshot.pt')
    features_path = os.path.join(curr_datapath, f'{ood_set}_features.pt')
    targets_path = os.path.join(curr_datapath, f'{ood_set}_targets.pt')
    captions_path = os.path.join(curr_datapath, f'{ood_set}_captions.txt')
    print(f"Loading CLIP with Vision Modul: {Config.VISION_MODEL}...")
    clip_model, preprocess = clip.load(Config.VISION_MODEL, device=Config.DEVICE)
    clip_model.eval()
    print("Loading GPT2 tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Initializing CaptionGenerator")
    caption_generator = CaptionGenerator(model_path=model_path,
                                         clip_model=clip_model,
                                         tokenizer=tokenizer,
                                         prefix_length=10)

    num_ood_classes = 10
    dataset = torchvision.datasets.CIFAR10(Config.DATAPATH,
                                           transform=preprocess)

    # random sample 10 classes
    # ood_classes_idx = random.sample(range(len(dataset.classes)), num_ood_classes)
    # ood_classes = [dataset.classes[i] for i in ood_classes_idx]
    # classes_left = [cl for cl in dataset.classes if cl not in ood_classes]
    #
    # print(f"Sampled {num_ood_classes} classes from dataset with "
    #       f"{len(dataset.classes)} classes. Sampled: {ood_classes_idx}.\n"
    #       f"Classes: {ood_classes}")
    #
    # # get the ood classes
    # oods = defaultdict(list)
    # for i, label in enumerate(dataset._labels):
    #     if label in ood_classes_idx:
    #         oods[label].append(dataset._images[i])
    #
    # ood_images, ood_labels = [], []
    # for label, images in oods.items():
    #     ood_images.extend(images)
    #     ood_labels.extend([label for _ in range(len(images))])
    # print(f"Got {len(ood_images)} images with {len(ood_labels)} labels")
    #
    # dataset._images = ood_images
    # dataset._labels = ood_labels
    #
    # # 1. remove the ood labels from the classnames
    # dataset.classes = classes_left
    #
    # # 2. features generation for left over classes
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

    print(f"Classifying {len(dataset.targets)} images")
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=8)
    ood_label = len(dataset.classes)
    features = []
    labels = []
    full_captions = []
    if not load_features:
        with torch.no_grad():
            for images, actual_targets in tqdm(dataloader):
                images = images.to(Config.DEVICE)
                targets = torch.tensor([ood_label for i in range(len(actual_targets))]).to(Config.DEVICE)
                images_features = clip_model.encode_image(images).to(Config.DEVICE, dtype=torch.float32)

                if generate_captions:
                    captions = generate_caption_with_generator(caption_generator, images_features, classnames,
                                                               cut_off_labels)
                    full_captions.extend(captions)
                images_features /= images_features.norm(dim=-1, keepdim=True)
                features.append(images_features)
                labels.append(targets)

            features = torch.cat(features)
            labels = torch.cat(labels)

        save_features_labels_captions(features, features_path, labels, targets_path, full_captions, captions_path)

    else:
        features, labels, full_captions = load_features_labels_captions(features_path, targets_path, captions_path)

    # full_logits = get_full_logits(features, zeroshot_weights, full_captions, clip_model, imagenet_templates)
    logits = 100. * features @ zeroshot_weights
    top1, top5, n = 0., 0., 0.
    acc1, acc5 = accuracy(logits, labels, top_k=(1, 5))
    top1 += acc1
    top5 += acc5

    n += features.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"\nClip Top1 Acc: {top1:.2f} with zeroshot")
    print(f"\nClip Top5 Acc: {top5:.2f} with zeroshot")

    print("Done")


if __name__ == '__main__':
    main()

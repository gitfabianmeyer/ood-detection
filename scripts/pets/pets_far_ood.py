import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import clip
import torch
import torchvision.datasets
from ood_detection.classnames import imagenet_templates
from ood_detection.config import Config

# init stuff
from ood_detection.datasets.stanfordcars import StandardizedStanfordCars
from ood_detection.models.dummy_zoc import CaptionGenerator
from ood_detection.ood_utils import zeroshot_classifier, \
    clean_caption, save_features_labels_captions, \
    load_features_labels_captions, get_full_logits, \
    generate_caption_with_generator, ood_accuracy

from tqdm import tqdm
from transformers import GPT2Tokenizer


def main():
    print("Starting run")
    load_features = False
    load_zeroshot = False
    batch_size = 16
    cut_off_labels = 5
    generate_captions = True

    ood_set = "ood_far_pets"
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
    id_dataset = torchvision.datasets.OxfordIIITPet(Config.DATAPATH,
                                                 transform=preprocess)

    # ood data
    ood_dataset = StandardizedStanfordCars(Config.DATAPATH,
                                           transform=preprocess)

    # 2. features generation for left over classes
    templates = imagenet_templates
    classnames = id_dataset.classes

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

    print(f"Classifying {len(ood_dataset._image_files)} images")
    dataloader = torch.utils.data.DataLoader(ood_dataset,
                                             batch_size=batch_size,
                                             num_workers=8)
    # add the ood label
    ood_label = len(id_dataset.classes)
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

    full_logits = get_full_logits(features, zeroshot_weights, full_captions, clip_model, imagenet_templates)

    top1, top5, n = 0., 0., 0.
    acc1, acc5 = ood_accuracy(full_logits, labels, len(id_dataset.classes), top_k=(1, 5))
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

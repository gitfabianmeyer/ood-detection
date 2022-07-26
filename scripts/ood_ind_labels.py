import os

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

from ood_detection.ood_utils import get_individual_ood_weights, zeroshot_classifier, accuracy

device = Config.DEVICE
print(f"Using {device}")
stopwords = set(stopwords.words('english'))
batch_size = 32


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


def clean_caption(caption, classnames):
    return [word for word in caption.split() if word not in classnames][:5]


def main(generate_caption=True):
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
    classnames = oxfordpets_classes
    dataset = torchvision.datasets.OxfordIIITPet(Config.DATAPATH,
                                                 transform=preprocess)
    print(f"Classifying {len(dataset._images)} images")
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=8)

    # get the label features
    zeroshot_weights = zeroshot_classifier(classnames, templates, clip_model)

    features = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            images_features = clip_model.encode_image(images).to(device, dtype=torch.float32)

            if generate_caption:
                captions = [caption_generator.generate_caption(img_feat, encoded=True) for img_feat in images_features]
                captions = [clean_caption(caption, classnames) for caption in captions]

            images_features /= images_features.norm(dim=-1, keepdim=True)
            features.append(images_features)
            labels.append(targets)

        # free space on cuda
        features = torch.cat(features).to('cpu')
        labels = torch.cat(labels).to('cpu')

    if torch.cuda.is_available():
        print("Trying to clear memory on CUDA")
        torch.cuda.empty_cache()
        features = features.to(device)
        labels = labels.to(device)
    # now: for each triple image | label | ood_label:
    # do: append ood_label to labels
    # do: classify


    full_logits = []
    for image, label, ood_labels in zip(features, labels, captions):
        ind_class_embeddings = get_individual_ood_weights(ood_labels,
                                                          clip_model,
                                                          templates=imagenet_templates)
        ind_zeroshot_weights = torch.cat([zeroshot_weights, ind_class_embeddings], dim=1).to(device,
                                                                                             dtype=torch.float32)

        # zeroshotting
        top1, top5, n = 0., 0., 0.
        logits = 100. * image @ ind_zeroshot_weights
        full_logits.append(logits)

    full_logits = torch.stack(full_logits)
    print(f"Shape should be: {len(dataset.classes) + 5} x {len(dataset._images)} and is: {full_logits.shape}")
    # no adapation needed, as new labels are always wrong
    acc1, acc5 = accuracy(full_logits, labels, top_k=(1, 5))
    top1 += acc1
    top5 += acc5

    n += features.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    print(f"\nClip Top1 Acc: {top1:.2f} with zeroshot")
    print(f"\nClip Top5 Acc: {top5:.2f} with zeroshot")


if __name__ == '__main__':
    print("Starting run")
    main()

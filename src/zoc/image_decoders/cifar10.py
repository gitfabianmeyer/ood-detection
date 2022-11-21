import os

from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import clip
import numpy as np
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer

from ood_detection.classnames import imagenet_templates
from ood_detection.ood_classification import get_dataset_features
from ood_detection.ood_utils import zeroshot_classifier, classify
from zoc.dataloaders.cifar10 import cifar10_single_isolated_class_loader, get_cifar10_loader
from zoc.utils import image_decoder


def classify_cifar10(model, preprocess):
    loader = get_cifar10_loader(preprocess)

    features, labels = get_dataset_features(loader, model, None, None)
    zeroshot_weights = zeroshot_classifier(loader.dataset.dataloaders, templates=imagenet_templates, clip_model=model)
    classify(features, zeroshot_weights, labels, "CIFAR10")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_path', type=str, default='/mnt/c/Users/fmeyer/Git/ZOC/trained_models/COCO/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.saved_model_path = os.path.join(args.trained_path, 'ViT-B32/')

    if not os.path.exists(args.saved_model_path):
        os.makedirs(args.saved_model_path)

    # initialize tokenizers for clip and bert, these two use different tokenizers
    bert_tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')

    clip_model, preprocess = clip.load('ViT-B/32')
    cliptokenizer = clip_tokenizer()

    bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    bert_config.is_decoder = True
    bert_config.add_cross_attention = True
    bert_model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                       config=bert_config).to(device).train()
    bert_model.load_state_dict(
        torch.load(args.saved_model_path + 'model_3.pt', map_location=torch.device(device))['net'])

    classify_cifar10(clip_model, preprocess)
    cifar10_loaders = cifar10_single_isolated_class_loader()
    runs = 5
    mean_list = []
    cifar10_labels = ['airplane', 'automobile', 'truck', 'horse', 'cat', 'bird', 'ship', 'deer', 'dog', 'frog']
    for i in tqdm(range(runs)):
        mean, _ = image_decoder(clip_model,
                                cliptokenizer,
                                bert_tokenizer,
                                bert_model,
                                device,
                                cifar10_labels,
                                image_loaders=cifar10_loaders)
        mean_list.append(mean)
    print(f"Scores for {runs} runs of 10 ablation splits: Mean: {np.mean(mean_list)}. Std: {np.std(mean_list)}")

import os

from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import argparse
import numpy as np

from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip
import torch
from transformers import BertGenerationTokenizer, BertGenerationConfig, BertGenerationDecoder

from ood_detection import classnames
from zoc.utils import image_decoder
from zoc.dataloaders.dtd_loader import dtd_single_isolated_class_loader, get_dtd_loader
from ood_detection.classnames import imagenet_templates
from ood_detection.ood_classification import get_dataset_features
from ood_detection.ood_utils import classify, zeroshot_classifier


def classify_dtd(model, preprocess):
    loader = get_dtd_loader(preprocess)

    features, labels = get_dataset_features(loader, model, None, None)
    zeroshot_weights = zeroshot_classifier(loader.dataset.dataloaders, templates=imagenet_templates, clip_model=model)
    classify(features, zeroshot_weights, labels, "DTD")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_path', type=str, default='/mnt/c/Users/fmeyer/Git/ZOC/trained_models/COCO/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.saved_model_path = os.path.join(args.trained_path,'ViT-B32/')

    if not os.path.exists(args.saved_model_path):
        os.makedirs(args.saved_model_path)

    # initialize tokenizers for clip and bert, these two use different tokenizers
    bert_tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')

    # clip_model = torch.jit.load(os.path.join('./trained_models', "{}.pt".format('ViT-B/32'))).to(device).eval()
    clip_model, preprocess = clip.load('ViT-B/32')
    cliptokenizer = clip_tokenizer()

    bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
    bert_config.is_decoder = True
    bert_config.add_cross_attention = True
    bert_model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder',
                                                       config=bert_config).to(device).train()
    bert_model.load_state_dict(
        torch.load(args.saved_model_path + 'model_3.pt', map_location=torch.device(device))['net'])

    dtd10_loaders = dtd_single_isolated_class_loader()
    classify_dtd(clip_model, preprocess)

    runs = 5
    mean_list = []

    for i in tqdm(range(runs)):
        mean, _ = image_decoder(clip_model,
                                cliptokenizer,
                                bert_tokenizer,
                                bert_model,
                                device,
                                classnames.dtd_classes,
                                image_loaders=dtd10_loaders)
        mean_list.append(mean)

    print(f"Scores for {runs} runs of 10 ablation splits: Mean: {np.mean(mean_list)}. Std: {np.std(mean_list)}")
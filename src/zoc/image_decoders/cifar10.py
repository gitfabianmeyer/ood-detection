import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import clip
from tqdm import tqdm
import numpy as np
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from sklearn.metrics import roc_auc_score

from ood_detection.classnames import imagenet_templates
from ood_detection.ood_classification import get_dataset_features
from ood_detection.ood_utils import zeroshot_classifier, classify
from zoc.dataloaders.cifar10 import cifar10_single_isolated_class_loader, get_cifar10_loader


def classify_cifar10(model, preprocess):
    loader = get_cifar10_loader(preprocess)

    features, labels = get_dataset_features(loader, model, None, None)
    zeroshot_weights = zeroshot_classifier(loader.dataset.classes, templates=imagenet_templates, clip_model=model)
    classify(features, zeroshot_weights, labels, "CIFAR10")


def image_decoder(clip_model, cliptokenizer, berttokenizer, device, image_loaders=None):
    splits = [['airplane', 'automobile', 'truck', 'horse', 'cat', 'bird', 'ship', 'deer', 'dog', 'frog'],
              ['airplane', 'automobile', 'truck', 'horse', 'cat', 'bird', 'ship', 'deer', 'dog', 'frog'],
              ['airplane', 'bird', 'deer', 'cat', 'horse', 'dog', 'ship', 'automobile', 'frog', 'truck'],
              ['dog', 'automobile', 'truck', 'ship', 'horse', 'airplane', 'bird', 'cat', 'deer', 'frog'],
              ['dog', 'horse', 'automobile', 'ship', 'deer', 'frog', 'airplane', 'truck', 'bird', 'cat'],
              ['ship', 'automobile', 'dog', 'cat', 'deer', 'frog', 'airplane', 'truck', 'bird', 'horse']]
    ablation_splits = [['airplane', 'automobile', 'truck', 'horse', 'cat', 'bird', 'ship', 'dog', 'deer', 'frog'],
                       ['airplane', 'automobile', 'truck', 'bird', 'ship', 'frog', 'deer', 'dog', 'horse', 'cat']]
    # ablation_splits = [['horse', 'cat', 'deer', 'frog'],
    #                   ['deer', 'frog', 'horse', 'cat']]

    auc_list_sum = []
    for split in ablation_splits:
        seen_labels = split[:8]
        print(f"seen: {seen_labels}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]
        targets = torch.tensor(8000 * [0] + 2000 * [1])
        max_num_entities = 0
        ood_probs_sum = []
        for i, semantic_label in enumerate(split):
            print(f"semantic label: {semantic_label}")
            loader = image_loaders[semantic_label]
            for idx, image in enumerate(tqdm(loader)):
                # if idx==10:break
                with torch.no_grad():
                    clip_out = clip_model.encode_image(image.to(device)).float()
                    clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)

                    # greedy generation
                    target_list, topk_list = greedysearch_generation_topk(clip_extended_embed)
                    print(f"target list {target_list}")
                    print(f"topk list {topk_list}")
                    target_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in target_list]
                    topk_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]

                    unique_entities = list(set(topk_tokens) - {semantic_label})
                    if len(unique_entities) > max_num_entities:
                        max_num_entities = len(unique_entities)
                    all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
                    all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

                    image_feature = clip_model.encode_image(image.to(device)).float()
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)
                    text_features = clip_model.encode_text(all_desc_ids.to(device)).float()
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    print(f"\nimage:{image_feature.shape}")
                    print(f"text {text_features.shape}")
                    zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()
                    print(f"zeroshot probs {zeroshot_probs.shape}")
                    break

                # detection score is accumulative sum of probs of generated entities
                ood_prob_sum = np.sum(zeroshot_probs[6:].detach().cpu().numpy())
                print(ood_prob_sum.shape)
                ood_probs_sum.append(ood_prob_sum)
                break
        auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))
        print('sum_ood AUROC={}'.format(auc_sum))
        auc_list_sum.append(auc_sum)
        break
    print('all auc scores:', auc_list_sum)
    print('auc sum', np.mean(auc_list_sum), np.std(auc_list_sum))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_path', type=str, default='/mnt/c/Users/fmeyer/Git/ZOC/trained_models/COCO/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.saved_model_path = os.path.join(args.trained_path, 'ViT-B32/')

    if not os.path.exists(args.saved_model_path):
        os.makedirs(args.saved_model_path)

    # initialize tokenizers for clip and bert, these two use different tokenizers
    berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')

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

    classify_cifar10(clip_model, preprocess)
    cifar10_loaders = cifar10_single_isolated_class_loader()
    image_decoder(clip_model, cliptokenizer, berttokenizer, device, image_loaders=cifar10_loaders)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



from ood_detection.classnames import imagenet_templates
from ood_detection.ood_classification import get_dataset_features
from ood_detection.ood_utils import classify, zeroshot_classifier

import argparse

from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip
import numpy as np
import torch
from ood_detection import classnames
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import BertGenerationTokenizer, BertGenerationConfig, BertGenerationDecoder

from zoc.utils import greedysearch_generation_topk, tokenize_for_clip, get_ablation_splits
from zoc.dataloaders.dtd_loader import dtd_single_isolated_class_loader, get_dtd_loader


def classify_dtd(model):
    loader = get_dtd_loader(preprocess)

    features, labels = get_dataset_features(loader, model, None, None)
    zeroshot_weights = zeroshot_classifier(loader.dataset.classnames, templates=imagenet_templates, clip_model=model)
    classify(features, zeroshot_weights, labels, "DTD")



def image_decoder(clip_model,
                  cliptokenizer,
                  berttokenizer,
                  device,
                  image_loaders=None,
                  id_classes=8,
                  ood_classes=2,
                  num_imgs_per_class=40):
    ablation_splits = get_ablation_splits(classnames.dtd_classes, n=3, id_classes=id_classes, ood_classes=ood_classes)

    auc_list_sum = []
    for split in ablation_splits:
        seen_labels = split[:id_classes]
        print(f"Seen labels: {seen_labels}")
        print(f"OOD Labels: {split[id_classes:]}")
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]
        # targets = torch.tensor(6000*[0] + 4000*[1])

        # see if you can set this manually
        targets = torch.tensor((num_imgs_per_class * id_classes) * [0] + (ood_classes*num_imgs_per_class) * [1])
        max_num_entities = 0 # ?
        ood_probs_sum = []
        for i, semantic_label in enumerate(split):
            loader = image_loaders[semantic_label]
            for idx, image in enumerate(tqdm(loader)):
                with torch.no_grad():

                    clip_out = clip_model.encode_image(image.to(device)).float()
                    clip_extended_embed = clip_out.repeat(1, 2).type(torch.FloatTensor)

                    # greedy generation
                    target_list, topk_list = greedysearch_generation_topk(clip_extended_embed,
                                                                          berttokenizer,
                                                                          bert_model,
                                                                          device)

                    topk_tokens = [berttokenizer.decode(int(pred_idx.cpu().numpy())) for pred_idx in topk_list]

                    unique_entities = list(set(topk_tokens) - {semantic_label})
                    if len(unique_entities) > max_num_entities:
                        max_num_entities = len(unique_entities)
                    all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
                    all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

                    image_feature = clip_out
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)
                    text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

                # detection score is accumulative sum of probs of generated entities
                ood_prob_sum = np.sum(zeroshot_probs[id_classes:].detach().cpu().numpy())
                ood_probs_sum.append(ood_prob_sum)
        auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))
        print('sum_ood AUROC={}'.format(auc_sum))
        auc_list_sum.append(auc_sum)
    print('all auc scores:', auc_list_sum)
    print('auc sum', np.mean(auc_list_sum), np.std(auc_list_sum))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_path', type=str, default='/home/fmeyer/ZOC/trained_models/COCO/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.saved_model_path = args.trained_path + '/ViT-B32/'

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
    bert_model.load_state_dict(torch.load(args.saved_model_path + 'model_3.pt',  map_location=torch.device(device))['net'])

    dtd10_loaders = dtd_single_isolated_class_loader()
    classify_dtd(clip_model, preprocess)
    image_decoder(clip_model, cliptokenizer, berttokenizer, device, image_loaders=dtd10_loaders)
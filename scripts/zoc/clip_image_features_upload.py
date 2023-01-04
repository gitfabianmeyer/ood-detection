import argparse
import copy
import logging
import os

import numpy as np
import sentencepiece
import pycocotools
import clip
import torch
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig, BertTokenizer

from torch.optim import AdamW

from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from clearml import Dataset, Task

task = Task.init(project_name="ma_fmeyer", task_name="clip_image_features")
task.execute_remotely('5e62040adb57476ea12e8593fa612186')
dataset_name = "COCO 2017 Dataset"
DATASET_PATH = Dataset.get(dataset_project='COCO-2017',
                           dataset_name=dataset_name
                           ).get_local_copy()
logger = task.get_logger()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {DEVICE}")


class MyCocoDetection:
    def __init__(self, root, train=True):
        self.filename = 'train2017' if train else 'val2017'
        self.root = root
        super(MyCocoDetection, self).__init__()

        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),  # 224 for vit, 288 for res50x4
            CenterCrop(224),  # 224 for vit, 288 for res50x4
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

        self.coco_dataset = CocoDetection(
            root=os.path.join(f'{self.root}/images', self.filename),
            annFile=os.path.join('{}/annotations'.format(self.root),
                                 'captions_{}.json'.format(self.filename)))

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, index):
        img = self.transform(self.coco_dataset[index][0])
        captions = self.coco_dataset[index][1]
        cap_list = []
        for num_captions, caption in enumerate(captions):
            if num_captions == 5:
                # print('more than 5 captions for this image', index)
                break
            cap = caption['caption']
            cap_list.append(cap)
        if len(cap_list) < 5:
            print('has less than 5 captions', index)
        return img, cap_list


def get_bert_training_features(coco_dataset, split, clip_backbone, tokenizer):
    sentences = get_bos_sentence_eos(coco_dataset, tokenizer, split, clip_backbone)
    return 1, 2, 3


def collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def get_clip_image_features(coco_dataset, split, clip_backbone, clip_model, torch_device):
    print('calculating all clip image encoder features')
    features_path = 'clip_image_features_{}_{}.npy'.format(split,
                                                           clip_backbone)

    print("Calculating clip image features")
    loader = DataLoader(dataset=coco_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    clip_out_all = []
    for i, (images, annot) in enumerate(tqdm(loader)):
        images = torch.stack(images)
        clip_out = clip_model.encode_image(images.to(torch_device))
        clip_out_all.append(clip_out.cpu().numpy())
    clip_out_all = np.concatenate(clip_out_all)

    try:
        np.save(os.path.join('/mnt/c/Users/fmeyer/Git/ood-detection/scripts/zoc/', features_path), arr=clip_out_all)
        'stored npy'
    except Exception as e:
        print(e)
    try:
        task.upload_artifact(name=features_path,
                             artifact_object=clip_out_all)
        print(f"Uploaded clip image features {split} as artifact")
    except Exception as e:
        print(e)
    return clip_out_all


def get_bos_sentence_eos(coco_dataset, berttokenizer, split, backbone):
    save_path = "bos_sentence_eos_{}_{}.npy".format(backbone, split)

    print('preprocessing all sentences...')
    bos_sentence_eos = []
    for i, (image, captions) in enumerate(tqdm(coco_dataset)):

        for caption in captions:
            bos_sentence_eos.append(berttokenizer.bos_token + ' ' + caption + ' ' + berttokenizer.eos_token)
    try:
        task.upload_artifact(name=save_path,
                             artifact_object=np.array(bos_sentence_eos))
        print(f"Uploaded {save_path} as artifact")
    except:
        print(f"Could store in {save_path}, continuing...")
    return bos_sentence_eos


def get_loader(train, clip_backbone, clip_model, berttokenizer, datapath):
    if train:
        split = 'train'
    else:
        split = 'val'

    coco_dataset = MyCocoDetection(root=datapath, train=train)
    clip_features = get_clip_image_features(coco_dataset, split, clip_backbone, clip_model, torch_device=DEVICE)

    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=25, help="End epoch")  # trained with 25 epochs
    parser.add_argument('--trained_path', type=str, default='./trained_models/COCO/')
    parser.add_argument('--bert_model', type=str, default='google/bert_for_seq_generation_L-24_bbc_encoder')
    parser.add_argument('--clip_vision', type=str, default='ViT-B/32')

    args = parser.parse_args()

    # initialize tokenizers for clip and bert, these two use different tokenizers
    if args.bert_model == "google/bert_for_seq_generation_L-24_bbc_encoder":
        berttokenizer = BertGenerationTokenizer.from_pretrained(args.bert_model)
    else:
        berttokenizer = BertTokenizer.from_pretrained(args.bert_model)

    cmodel, _ = clip.load(args.clip_vision)

    for split in [True, False]:
        tloader = get_loader(train=split, clip_backbone=args.clip_vision, clip_model=cmodel,
                             berttokenizer=berttokenizer, datapath=DATASET_PATH)
    print("Finished")

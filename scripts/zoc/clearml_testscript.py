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

run_clearml = True
load_from_clearml = True
if run_clearml:
    task = Task.init(project_name="ma_fmeyer", task_name="Train Decoder")
    task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    dataset_name = "COCO 2017 Dataset"
    DATASET_PATH = Dataset.get(dataset_project='COCO-2017',
                               dataset_name=dataset_name
                               ).get_local_copy()
    logger = task.get_logger()

else:
    DATASET_PATH = '/mnt/c/users/fmeyer/git/ood-detection/data/coco'

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


@torch.no_grad()
def eval_decoder(bert_model, loader):
    num_batch = len(iter(loader))
    print('evaluating loss on validation data ...')
    acc_loss = 0.
    bert_model.eval()
    for _, batch in enumerate(tqdm(loader)):
        input_ids, attention_mask, label_ids, clip_embeds = batch
        clip_extended_embed = clip_embeds.repeat(1, 2).type(torch.FloatTensor)

        N, seq_length = input_ids.shape
        position_ids = torch.arange(0, seq_length).expand(N, seq_length)
        out = bert_model(input_ids=input_ids.to(DEVICE),
                         position_ids=position_ids.to(DEVICE),
                         attention_mask=attention_mask.to(DEVICE),
                         encoder_hidden_states=clip_extended_embed.unsqueeze(1).to(DEVICE),
                         labels=label_ids.to(DEVICE))
        acc_loss += out.loss.detach().item()

    print('Average loss on {} validation batches={}\n'.format(num_batch, acc_loss / num_batch))
    return acc_loss


def train_decoder(bert_model, train_loader, eval_loader, optimizer):
    early_stop = 0
    num_batch = len(iter(train_loader))
    print(f"Starting training for max {args.num_epochs} epochs...")

    best_val_loss = np.inf
    for epoch in range(1, args.num_epochs + 1):
        print('Training : epoch {}'.format(epoch))
        acc_loss = 0.

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            if batch_idx == 1:
                print(f"Seems to run, breaking now")
                break
            input_ids, attention_mask, label_ids, clip_embeds = batch
            clip_extended_embed = clip_embeds.repeat(1, 2).type(torch.FloatTensor)

            N, seq_length = input_ids.shape
            position_ids = torch.arange(0, seq_length).expand(N, seq_length)
            bert_model.train()
            out = bert_model(input_ids=input_ids.to(DEVICE),
                             position_ids=position_ids.to(DEVICE),
                             attention_mask=attention_mask.to(DEVICE),
                             encoder_hidden_states=clip_extended_embed.unsqueeze(1).to(DEVICE),
                             labels=label_ids.to(DEVICE))

            out.loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()

            curr_loss=out.loss.detach().item()
            acc_loss += curr_loss
            if run_clearml:
                logger.report_scalar("train", "loss",
                                     iteration=(epoch * num_batch + batch_idx),
                                     value=curr_loss)

        validation_loss = eval_decoder(bert_model, eval_loader)
        if run_clearml:
            logger.report_scalar("val", "loss",
                                 iteration=epoch,
                                 value=validation_loss)

        print('validation loss in this epoch: ', validation_loss)
        state = {'net': bert_model.state_dict(),
                 'epoch': epoch,
                 'validation loss': validation_loss}

        if epoch == 1:
            best_val_loss = validation_loss
            torch.save(state, 'model_dump.pt')
        else:
            if validation_loss < best_val_loss:
                early_stop = 0
                best_val_loss = validation_loss
                torch.save(state, 'model.pt')
            else:
                early_stop += 1

        print('Average loss on {} training batches in this epoch:{}\n'.format(num_batch, acc_loss / num_batch))

        if early_stop >= 4:
            print(f"No improvements on val data for {early_stop} iterations. Breaking now")
            break

    return acc_loss


def get_bert_training_features(coco_dataset, split, clip_backbone, tokenizer):
    sentences = get_bos_sentence_eos(coco_dataset, tokenizer, split, clip_backbone)
    print(f'tokenizing all processed sentences for {split}...')
    tokenized = tokenizer(sentences, padding=True,
                          truncation=True, max_length=77,
                          return_token_type_ids=False, return_tensors='np')

    label_ids = copy.deepcopy(tokenized['input_ids'])
    label_ids[label_ids == 0] = -100
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    return input_ids, attention_mask, label_ids


def collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def get_clip_image_features(coco_dataset, split, clip_backbone, clip_model, torch_device):
    print('calculating all clip image encoder features')
    features_path = 'clip_image_features_{}_{}.npy'.format(split,
                                                           clip_backbone)

    if load_from_clearml:
        artifact_task = Task.get_task(project_name='ma_fmeyer', task_name='clip_image_features')
        artifact = artifact_task.artifacts[features_path].get_local_copy()
        artifact = np.load(artifact)
        clip_out_all = artifact[features_path]

    else:
        print("Calculating clip image features")
        loader = DataLoader(dataset=coco_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
        clip_out_all = []
        for i, (images, annot) in enumerate(tqdm(loader)):
            images = torch.stack(images)
            clip_out = clip_model.encode_image(images.to(torch_device))
            clip_out_all.append(clip_out.cpu().numpy())
        clip_out_all = np.concatenate(clip_out_all)

        try:
            if run_clearml:
                task.upload_artifact(name=features_path,
                                     artifact_object=clip_out_all)
                print(f"Uploaded clip image features {split} as artifact")
        except:
            print(f"Couldn't store image features.")

    return clip_out_all


def get_bos_sentence_eos(coco_dataset, berttokenizer, split, backbone):
    features_path = "bos_sentence_eos_{}_{}.npy".format(backbone, split)
    if load_from_clearml:
        artifact_task = Task.get_task(project_name='ma_fmeyer', task_name='bos_sentence_eos')
        artifact = artifact_task.artifacts[features_path].get_local_copy()
        artifact = np.load(artifact)
        bos_sentence_eos = artifact[features_path]
        bos_sentence_eos = bos_sentence_eos.tolist()

    else:
        print('preprocessing all sentences...')
        bos_sentence_eos = []
        for i, (image, captions) in enumerate(tqdm(coco_dataset)):

            for caption in captions:
                bos_sentence_eos.append(berttokenizer.bos_token + ' ' + caption + ' ' + berttokenizer.eos_token)
        try:
            if run_clearml:

                task.upload_artifact(name=features_path,
                                     artifact_object=np.array(bos_sentence_eos))
                print(f"Uploaded {features_path} as artifact")
        except:
            print(f"Couldn't store in {features_path}, continuing...")
    return bos_sentence_eos


def get_loader(train, clip_backbone, clip_model, berttokenizer, datapath):
    if train:
        split = 'train'
    else:
        split = 'val'

    coco_dataset = MyCocoDetection(root=datapath, train=train)
    clip_features = get_clip_image_features(coco_dataset, split, clip_backbone, clip_model, torch_device=DEVICE)
    input_ids, attention_mask, label_ids = get_bert_training_features(coco_dataset, split, clip_backbone, berttokenizer)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    clip_features = torch.tensor(clip_features, dtype=torch.long)
    print(input_ids.size(), attention_mask.size(), label_ids.size(), clip_features.size())
    hidden_size = clip_features.size(1)
    print(clip_features.repeat(1, 5).view(-1, hidden_size).size())
    dataset = TensorDataset(input_ids, attention_mask, label_ids, clip_features.repeat(1, 5).view(-1, hidden_size))
    loader = DataLoader(dataset=dataset, batch_size=128, num_workers=2, shuffle=True)
    return loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=2, help="End epoch")  # trained with 25 epochs
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
    tloader = get_loader(train=True, clip_backbone=args.clip_vision, clip_model=cmodel,
                         berttokenizer=berttokenizer, datapath=DATASET_PATH)
    eloader = get_loader(train=False, clip_backbone=args.clip_vision, clip_model=cmodel,
                         berttokenizer=berttokenizer, datapath=DATASET_PATH)

    bert_config = BertGenerationConfig.from_pretrained(args.bert_model)
    bert_config.is_decoder = True
    bert_config.add_cross_attention = True
    bmodel = BertGenerationDecoder.from_pretrained(args.bert_model,
                                                   config=bert_config).to(DEVICE).train()

    optimizer = AdamW(bmodel.parameters(), lr=args.lr)

    loss = train_decoder(bmodel, tloader, eloader, optimizer)
    print('final training loss={}'.format(loss))

import logging
import os.path
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import clip
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ood_detection.classification_utils import zeroshot_classifier, get_dataset_features
from ood_detection.config import Config

from zeroshot.classification import get_cosine_similarity_matrix_for_normed_features
from zeroshot.utils import get_normalized_image_features

_logger = logging.getLogger()
device = Config.DEVICE


class WeightAdapter(nn.Module):
    def __init__(self,
                 cache_keys):
        super(WeightAdapter, self).__init__()
        self.linear1 = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(torch.float32)
        self.weight = nn.Parameter(cache_keys.t())

    def forward(self, x):
        return self.linear1(x)


def zeroshot(clip_logits, test_labels):
    return get_acc_f1(clip_logits, test_labels)


def get_acc_f1_for_adapter(image_features, cache_keys, cache_values, clip_logits, targets, alpha, beta, adapter=None):
    if adapter:
        affinity = adapter(image_features)
    else:
        affinity = image_features @ cache_keys
    cache_logits_test = get_cache_logits(affinity, cache_values, beta)
    tip_logits = clip_logits + cache_logits_test * alpha
    acc_tip, f1_tip = get_acc_f1(tip_logits, targets)
    return acc_tip, f1_tip


def full_clip_tip_classification(dataset, kshots, train_epochs, init_alpha, init_beta, lr, eps, augment_epochs,
                                 temperature):
    _logger.info("Initializing everything...")
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    clip_model.eval()
    train_set = get_tip_adapter_train_set(dataset, kshots)
    _logger.info(f"len trainset: {len(train_set)}. Should be: {len(train_set.classes) * kshots} (max)")

    # run everything on the val set first.
    cache_keys, cache_values = get_cache_model(train_set, clip_model, augment_epochs=augment_epochs)

    val_features, val_labels, label_features, classes = get_dataset_features_with_split(dataset, clip_model,
                                                                                        clip_transform, 'val')

    test_features, test_labels, _, _ = get_dataset_features_with_split(dataset, clip_model,
                                                                       clip_transform,
                                                                       'test')
    # zeroshot
    return run_full_tip_from_features(cache_keys, cache_values, clip_model, eps, init_alpha, init_beta, label_features,
                                      lr, temperature, test_features, test_labels, train_epochs, train_set,
                                      val_features, val_labels)


def run_full_tip_from_features(cache_keys, cache_values, clip_model, eps, init_alpha, init_beta, label_features, lr,
                               temperature, test_features, test_labels, train_epochs, train_set, val_features,
                               val_labels):
    clip_logits_val = get_cosine_similarity_matrix_for_normed_features(val_features, label_features, temperature)
    val_zsa, val_f1 = get_acc_f1(clip_logits_val, val_labels)
    _logger.info(f'CLIP ZEROSHOT ACCURACY: {val_zsa:.3f}\tF1: {val_f1:.3f}')
    # tip adapter
    tip_best_alpha, tip_best_beta = run_tip_adapter(val_features,
                                                    val_labels,
                                                    label_features,
                                                    cache_keys,
                                                    cache_values,
                                                    clip_logits_val,
                                                    init_alpha,
                                                    init_beta,
                                                    temperature)
    tipf_best_alpha, tipf_best_beta, adapter = run_tip_adapter_finetuned(train_set, clip_model,
                                                                         val_features, val_labels,
                                                                         label_features, cache_keys,
                                                                         cache_values, init_alpha,
                                                                         init_beta, train_epochs,
                                                                         lr, eps,
                                                                         temperature)
    # load test features, the adapter with weights, and run everything
    _logger.info("Evaluation on test set...")
    # zeroshot
    clip_logits_test = get_cosine_similarity_matrix_for_normed_features(test_features, test_labels, temperature)
    zsa, f1 = get_acc_f1(clip_logits_test, test_labels)
    # tip
    acc_tip, f1_tip = get_acc_f1_for_adapter(test_features, cache_keys, cache_values, clip_logits_test, test_labels,
                                             tip_best_alpha, tip_best_beta)
    # tipf
    acc_tipf, f1_tipf = get_acc_f1_for_adapter(test_features, cache_keys, cache_values, clip_logits_test, test_labels,
                                               tipf_best_alpha, tipf_best_beta, adapter)
    results = {"ZEROSHOT": zsa, "zf1": f1, "TIP ADAPTER": acc_tip, "TIP F1": f1_tip,
               "TIP-F ADAPTER": acc_tipf, "TIP-F F1": f1_tipf,
               "tip_best_alpha": tip_best_alpha, "tip_best_beta": tip_best_beta,
               "tipf_best_alpha": tipf_best_alpha, "tipf_best_beta": tipf_best_beta}
    return results


def get_label_dict(train_images):
    split_by_label_dict = defaultdict(list)

    # build kshot set
    for i in range(len(train_images)):
        split_by_label_dict[int(train_images.targets[i])].append(train_images.data[i])

    return split_by_label_dict


def get_truncated_to_min(label_dict, kshots):
    shortest = min([len(length) for length in label_dict.values()])
    if shortest < kshots:
        kshots = shortest
        _logger.warning(f"Set kshots to min class len: {shortest}")

    imgs, targets = [], []
    for label, items in label_dict.items():
        imgs = imgs + random.sample(items, kshots)
        targets = targets + [label for _ in range(kshots)]
    _logger.info(F"Truncated: {len(imgs), len(targets)}")
    return imgs, targets


def get_kshot_set(dataset, kshots):
    label_dict = get_label_dict(dataset)
    imgs, targets = get_truncated_to_min(label_dict, kshots)

    dataset.data = imgs
    dataset.targets = np.array(targets).squeeze()
    return dataset


def get_tip_adapter_train_set(dataset, kshots):
    _logger.info("Creating train set")
    train_transform = get_train_transform()
    dataset = dataset(data_path=Config.DATAPATH,
                      split='train',
                      transform=train_transform)

    return get_kshot_set(dataset, kshots)


@torch.no_grad()
def get_cache_model(train_set, model, augment_epochs=10):
    _logger.info(f'Getting train features aggregated...')

    train_loader = DataLoader(train_set,
                              batch_size=512,
                              num_workers=1,
                              shuffle=False)

    cache_values = []
    train_images_features_agg = []

    for augment_idx in range(augment_epochs):

        _logger.info(f"Augmenting features {augment_idx}/{augment_epochs}")
        train_images_features = []

        for i, (images, targets) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            images_features = model.encode_image(images)
            train_images_features.append(images_features)

            if augment_idx == 0:
                targets = targets.to(device)

                cache_values.append(targets)

        images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
        train_images_features_agg.append(images_features_cat)

    train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(0)
    train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
    train_images_features_agg = train_images_features_agg.permute(1, 0)

    cache_values = F.one_hot(torch.cat(cache_values, dim=0))
    cache_values = cache_values.to(torch.float32)
    cache_keys = train_images_features_agg.to(torch.float32)
    assert cache_keys.shape[1] == cache_values.shape[0], f"ck shape:{cache_keys.shape}, cv {cache_values.shape[0]}"
    assert cache_values.shape[1] == len(
        train_set.classes), f"cv {cache_values.shape[0]}, tain set: {len(train_set.classes)}"
    return cache_keys.to(torch.float32), cache_values.to(torch.float32)


def create_tip_train_set(dset, seen_labels, kshots, split='train'):
    dataset = get_dataset_with_shorted_classes(dset, seen_labels, split)

    dataset = get_kshot_set(dataset, kshots)
    return dataset


def get_dataset_with_shorted_classes(dset, seen_labels, split):
    dataset = dset(Config.DATAPATH,
                   transform=get_train_transform(),
                   split=split)
    _logger.info(f"Creating {split} set for the seen labels")
    new_class_to_idx = {seen_labels[i]: i for i in range(len(seen_labels))}
    new_idx_to_class = {value: key for (key, value) in new_class_to_idx.items()}
    new_images, new_targets = [], []
    for image, target in zip(dataset.data, dataset.targets):
        old_label = dataset.idx_to_class[int(target)]
        if old_label in seen_labels:
            # get only seen images & new labels for them
            new_images.append(image)
            new_targets.append(new_class_to_idx[old_label])
    dataset.data = new_images
    dataset.targets = np.array(new_targets)
    dataset.idx_to_class = new_idx_to_class
    dataset.class_to_idx = new_class_to_idx
    dataset.classes = seen_labels
    return dataset


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224,
                                     scale=(0.5, 1),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])


@torch.no_grad()
def get_dataset_features_from_dataset_with_split(dataset, model):
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=1)
    features, labels = get_dataset_features(dataloader, model)
    label_features = zeroshot_classifier(dataset.classes, dataset.templates, model)
    classes = dataset.classes

    return features, labels, label_features, classes


@torch.no_grad()
def get_dataset_features_with_split(dataset, model, transform, split):
    dataset = dataset(data_path=Config.DATAPATH,
                      split=split,
                      transform=transform)
    return get_dataset_features_from_dataset_with_split(dataset, model)


def get_cache_logits(affinity, cache_values, beta):
    cache_logits = ((-1) * (beta - beta * affinity.to(torch.float32))).exp() @ cache_values
    return cache_logits.to(Config.DEVICE)


def get_acc_f1(logits, test_labels):
    logits_topk = logits.topk(1, 1, True, True)[1].t().squeeze()
    acc = accuracy_score(test_labels.cpu().numpy(), logits_topk.cpu().numpy()) * 100
    f1 = f1_score(test_labels.cpu().numpy(), logits_topk.cpu().numpy(), average='macro') * 100
    return acc, f1


def search_hp(cache_keys, cache_values, features, labels, zeroshot_weights, temperature, adapter=None):
    best_acc = 0
    best_beta, best_alpha = 0, 0

    if adapter:
        affinity = adapter(features)
    else:
        affinity = features @ cache_keys

    for beta in np.linspace(0.1, 5, 10):
        for alpha in np.linspace(.1, 5, 10):

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

            clip_logits = get_cosine_similarity_matrix_for_normed_features(features, zeroshot_weights, temperature)
            tip_logits = clip_logits + cache_logits * alpha
            acc, _ = get_acc_f1(tip_logits, labels)  # eval only on acc

            if acc > best_acc:
                _logger.info("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha

    _logger.info("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def run_tip_adapter(val_features, val_labels, zeroshot_weights, cache_keys, cache_values, clip_logits, alpha,
                    beta, temperature):

    _logger.info(f"Running TIP Adapter - NO FINETUNING")
    # n_images * feature_size @ (num_classes * feature_size).t() --> n_images x num_classes
    affinity = val_features @ cache_keys
    cache_logits = get_cache_logits(affinity, cache_values, beta)
    tip_logits = clip_logits + cache_logits * alpha

    acc, f1 = get_acc_f1(tip_logits, val_labels)
    _logger.info(f'CLIP ZEROSHOT ACCURACY: {acc:.3f}\tF1: {f1:.3f}')

    best_beta, best_alpha = search_hp(cache_keys=cache_keys,
                                      cache_values=cache_values,
                                      features=val_features,
                                      labels=val_labels,
                                      zeroshot_weights=zeroshot_weights,
                                      temperature=temperature)
    return best_alpha, best_beta


def run_tip_adapter_finetuned(train_set, model,
                              val_features, val_labels,
                              zeroshot_weights, cache_keys,
                              cache_values, train_epochs, lr,
                              eps, temperature):
    _logger.info(f"Running TIP Adapter - FINETUNING")

    # set init residual ratio to 1 ( new & old knowledge balanced)
    init_alpha = 1.
    # set sharpness nearly balanced
    init_beta = 1.17

    train_loader_shuffle = DataLoader(train_set,
                                      batch_size=128,
                                      shuffle=True,
                                      num_workers=2)

    adapter, optimizer, scheduler = init_adapter(cache_keys, eps, lr, train_epochs, train_loader_shuffle)

    best_acc, best_epoch, best_f1 = 0, 0, 0
    losses, learning_rates, accuracies = [], [], []

    # finetune and store everything
    for epoch in tqdm(range(train_epochs)):
        _logger.info(f"Training epoch\t{epoch}/{train_epochs}")
        adapter.train()

        batch_losses = []

        for i, (images, targets) in enumerate(train_loader_shuffle):
            image_features = get_normalized_image_features(model, images)

            affinity = adapter(image_features.to(torch.float32))
            cache_logits = get_cache_logits(affinity, cache_values, init_beta)
            clip_logits = get_cosine_similarity_matrix_for_normed_features(image_features, zeroshot_weights,
                                                                           temperature)
            tip_logits = clip_logits + cache_logits * init_alpha

            loss = F.cross_entropy(tip_logits, targets.to(device))
            batch_losses.append(loss.detach().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        losses.append(sum(batch_losses))
        learning_rates.append(current_lr)
        _logger.info(f"LOSS: {sum(batch_losses)}, LR: {current_lr}")

        # eval on val set
        adapter.eval()
        affinity = adapter(val_features)
        cache_logits = get_cache_logits(affinity,
                                        cache_values,
                                        init_beta)
        clip_logits = get_cosine_similarity_matrix_for_normed_features(val_features, zeroshot_weights, temperature)
        tip_logits = clip_logits + cache_logits * init_alpha
        acc, f1 = get_acc_f1(tip_logits, val_labels)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            _logger.info(f"New best acc: {best_acc:.3f} \t(f1: {best_f1:.3f})")
            best_adapter = adapter
            store_adapter(adapter, train_set.name)

    # search the best alpha and beta
    _logger.info("Loading best state dict for hp search....")
    adapter.load_state_dict(load_adapter(train_set.name))
    adapter.eval()

    best_beta, best_alpha = search_hp(cache_keys=cache_keys,
                                      cache_values=cache_values,
                                      features=val_features,
                                      labels=val_labels,
                                      zeroshot_weights=zeroshot_weights,
                                      adapter=adapter,
                                      temperature=temperature)
    return best_alpha, best_beta, best_adapter.eval()


def init_adapter(cache_keys, eps, lr, train_epochs, train_loader_shuffle):
    adapter = WeightAdapter(cache_keys).to(device)
    adapter.weight = nn.Parameter(cache_keys.t())
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs * len(train_loader_shuffle))
    return adapter, optimizer, scheduler


def store_adapter(model, dataset):
    adapter_path = os.path.join(Config.DATAPATH, 'tip-adapter', dataset)
    os.makedirs(adapter_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(adapter_path, 'tip_adapter.pt'))


def load_adapter(dataset):
    adapter_path = os.path.join(Config.DATAPATH, 'tip-adapter', dataset, 'tip_adapter.pt')
    return torch.load(adapter_path, map_location=Config.DEVICE)


def load_hyperparams_from_training(name):
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs('thesis-tip-adapters-16_shots-test')
    for run in runs:
        if run.name == name:
            set_run = run
            break
    df = pd.DataFrame(set_run.history())

    tipf_best_beta = 'tipf_best_beta'
    tipf_best_alpha = 'tipf_best_alpha'
    tip_best_beta = 'tip_best_beta'
    tip_best_alpha = 'tip_best_alpha'
    return {tip_best_alpha: df.iloc[0][tip_best_alpha],
            tip_best_beta: df.iloc[0][tip_best_beta],
            tipf_best_alpha: df.iloc[0][tipf_best_alpha],
            tipf_best_beta: df.iloc[0][tipf_best_beta]}

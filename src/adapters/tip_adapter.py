import logging
import os.path
from collections import defaultdict
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import clip
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ood_detection.classification_utils import zeroshot_classifier
from ood_detection.config import Config

_logger = logging.getLogger()
device = Config.DEVICE


class WeightAdapter(nn.Module):
    def __init__(self,
                 cache_keys):
        super(WeightAdapter, self).__init__()
        self.linear1 = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(torch.float32)
        self.weight = nn.Parameter(cache_keys.t())
        store_adapter(self, 'test')

    def forward(self, x):
        return self.linear1(x)


def zeroshot(clip_logits, test_labels):
    return get_acc_f1(clip_logits, test_labels)


def get_adapter_weights(train_set, test_set, model, train_epoch=1, alpha=1., beta=1.17, lr=0.001, eps=1e-4):
    _logger.info("Initializing everything...")
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    clip_model.eval()
    cache_keys, cache_values = get_cache_model(train_set, clip_model, augment_epochs=10)

    test_features, test_labels, label_features, classes = get_test_features_tip(test_set, clip_model, clip_transform)

    _logger.info(f"Running TIP Adapter - FINETUNING")

    train_loader_shuffle = DataLoader(train_set,
                                      batch_size=256,
                                      shuffle=True,
                                      num_workers=1)
    adapter = WeightAdapter(cache_keys).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch * len(train_loader_shuffle))

    best_acc, best_epoch, best_f1 = 0, 0, 0
    losses, learning_rates, accuracies = [], [], []
    for epoch in range(train_epoch):
        _logger.info(f"Training epoch\t{epoch}/{train_epoch}")
        adapter.train()

        batch_losses = []

        for i, (images, targets) in enumerate(tqdm(train_loader_shuffle)):
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features.to(torch.float32))

            cache_logits = get_cache_logits(affinity, cache_values, beta)

            clip_logits = 100. * image_features.to(torch.float32) @ label_features.t()
            clip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(clip_logits, targets)
            batch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        losses.append(sum(batch_losses))
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        _logger.info(f"LOSS: {sum(batch_losses)}, LR: {current_lr}")

        # eval
        adapter.eval()
        affinity = adapter(test_features)
        cache_logits = get_cache_logits(affinity, cache_values, beta)
        clip_logits = 100. * test_features @ label_features.t()
        tip_logits = clip_logits + cache_logits * alpha
        acc, f1 = get_acc_f1(tip_logits, test_labels)
        if acc > best_acc:
            best_acc = acc
            _logger.info(f"New best acc: {acc:.3f} (f1: {f1:.3f})")
            # best_epoch = epoch
            finetuned_adapter_weights = adapter.weight  # maybe return them

    return finetuned_adapter_weights


def clip_tip_adapter(dataset, kshots, train_epochs, init_alpha, init_beta, lr, eps, augment_epochs):
    _logger.info("Initializing everything...")
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    clip_model.eval()
    train_set = get_tip_adapter_train_set(dataset, kshots)
    _logger.info(f"len trainset: {len(train_set)}. Should be: {len(train_set.classes) * kshots} (max)")

    # run everything on the val set first.
    _logger.info('----- VALIDATION PHASE-------')
    cache_keys, cache_values = get_cache_model(train_set, clip_model, augment_epochs=augment_epochs)

    val_features, val_labels, label_features, classes = get_dataset_features_with_split(dataset, clip_model, clip_transform, 'val')

    # zeroshot
    clip_logits_val = 100. * val_features @ label_features.t()
    val_zsa, val_f1 = zeroshot(clip_logits_val, val_labels)
    _logger.info(f'CLIP ZEROSHOT ACCURACY: {val_zsa:.3f}\tF1: {val_f1:.3f}')

    # tip adapter
    tip_best_alpha, tip_best_beta = run_tip_adapter(val_features,
                                                    val_labels,
                                                    label_features,
                                                    cache_keys,
                                                    cache_values,
                                                    clip_logits_val,
                                                    init_alpha,
                                                    init_beta)

    tipf_best_alpha, tipf_best_beta = run_tip_adapter_finetuned(train_set, clip_model,
                                                                val_features, val_labels,
                                                                label_features, cache_keys,
                                                                cache_values, init_alpha,
                                                                init_beta, train_epochs,
                                                                lr, eps)

    # load test features, the adapter with weights, and run everything

    _logger.info("Evaluation on test set...")
    test_features, test_labels, label_features, classes = get_dataset_features_with_split(dataset, clip_model, clip_transform,
                                                                               'test')
    # zeroshot
    clip_logits_test = 100. * test_features @ label_features.t()
    zsa, f1 = zeroshot(clip_logits_test, test_labels)

    # tip
    affinity = test_features @ cache_keys
    cache_logits_test = get_cache_logits(affinity, cache_values, tip_best_beta)
    tip_logits = clip_logits_test + cache_logits_test * tip_best_alpha
    acc_tip_no, f1_tip_no = zeroshot(tip_logits, test_labels)

    # tipf
    adapter = WeightAdapter(cache_keys).to(device)
    adapter.load_state_dict(load_adapter(train_set.name))
    affinity = adapter(test_features)
    cache_logits_test = get_cache_logits(affinity, cache_values, tipf_best_beta)
    tipf_logits = clip_logits_test + cache_logits_test * tipf_best_alpha
    acc_tip_fine, f1_tip_fine = zeroshot(tipf_logits, test_labels)

    results = {"ZEROSHOT": zsa, "zf1": f1, "TIP ADAPTER": acc_tip_no, "TIP F1": f1_tip_no,
               "TIP-F ADAPTER": acc_tip_fine, "TIP-F F1": f1_tip_fine,
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
    shortest = min([len(l) for l in label_dict.values()])
    if shortest < kshots:
        kshots = shortest
        _logger.warning(f"Set kshots to min class len: {shortest}")

    imgs, targets = [], []
    for label, items in label_dict.items():
        imgs = imgs + random.sample(items, kshots)
        targets = targets + [label for _ in range(kshots)]
    _logger.info(F"Truncated: {len(imgs), len(targets)}")
    return imgs, targets


def get_kshot_train_set(dataset, kshots):
    label_dict = get_label_dict(dataset)
    imgs, targets = get_truncated_to_min(label_dict, kshots)

    dataset.data = imgs
    dataset.targets = targets
    return dataset


def get_tip_adapter_train_set(dataset, kshots):
    _logger.info("Creating train set")
    train_transform = get_train_transform()
    dataset = dataset(data_path=Config.DATAPATH,
                      split='train',
                      transform=train_transform)

    return get_kshot_train_set(dataset, kshots)


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
                cache_values.append(int(targets))

        images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
        train_images_features_agg.append(images_features_cat)

    train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(0)
    train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
    train_images_features_agg = train_images_features_agg.permute(1, 0)

    cache_values = F.one_hot(torch.cat(cache_values, dim=0))
    cache_values = cache_values.to(torch.float32)
    cache_keys = train_images_features_agg.to(torch.float32)
    assert cache_keys.shape[1] == cache_values.shape[0], f"ck shape:{cache_keys.shape}, cv {cache_values.shape[0]}"
    assert cache_values.shape[1] == len(train_set.classes), f"cv {cache_values.shape[0]}, tain set: {len(train_set.classes)}"
    return cache_keys, cache_values


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
def get_test_features_tip(dataset, model):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1)
    test_features, test_labels = [], []

    _logger.info("Getting test features...")
    for idx, (images, targets) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        targets = targets.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        test_features.append(image_features)
        test_labels.append(targets)
    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)

    test_features = test_features.to(torch.float32)
    test_labels = test_labels.to(torch.float32)
    label_features = zeroshot_classifier(dataset.classes, dataset.templates, model).to(torch.float32)
    classes = dataset.classes

    return test_features, test_labels, label_features, classes


@torch.no_grad()
def get_dataset_features_with_split(dataset, model, transform, split):
    dataset = dataset(data_path=Config.DATAPATH,
                      split=split,
                      transform=transform)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=1)
    test_features, test_labels = [], []

    _logger.info(f"Getting {split} features...")
    for idx, (images, targets) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        targets = targets.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        test_features.append(image_features)
        test_labels.append(targets)
    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)

    test_features = test_features.to(torch.float32)
    test_labels = test_labels.to(torch.float32)
    label_features = zeroshot_classifier(dataset.classes, dataset.templates, model).to(torch.float32)
    classes = dataset.classes

    return test_features, test_labels, label_features, classes


def get_cache_logits(affinity, cache_values, beta):
    return ((-1) * (beta - beta * affinity.to(torch.float32))).exp() @ cache_values


def get_acc_f1(logits, test_labels):
    logits_topk = logits.topk(1, 1, True, True)[1].t().squeeze()
    acc = accuracy_score(test_labels.cpu().numpy(), logits_topk.cpu().numpy()) * 100
    f1 = f1_score(test_labels.cpu().numpy(), logits_topk.cpu().numpy(), average='macro') * 100
    return acc, f1


def search_hp(cache_keys, cache_values, features, labels, clip_weights, adapter=None):
    best_acc = 0
    best_beta, best_alpha = 0, 0

    if adapter:
        affinity = adapter(features)
    else:
        affinity = features @ cache_keys

    for beta in np.linspace(0.1, 5, 10):
        for alpha in np.linspace(.1, 5, 10):

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * features @ clip_weights.t()
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
                    beta):
    # first: simply go on val set
    # second: eval alpha and beta

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
                                      clip_weights=zeroshot_weights)
    return best_alpha, best_beta


def run_tip_adapter_finetuned(train_set, model,
                              val_features, val_labels,
                              zeroshot_weights, cache_keys,
                              cache_values, alpha, beta,
                              train_epochs, lr,
                              eps):
    _logger.info(f"Running TIP Adapter - FINETUNING")

    train_loader_shuffle = DataLoader(train_set,
                                      batch_size=128,
                                      shuffle=True,
                                      num_workers=2)

    adapter = WeightAdapter(cache_keys).to(device)
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs * len(train_loader_shuffle))

    best_acc, best_epoch, best_f1 = 0, 0, 0
    losses, learning_rates, accuracies = [], [], []

    # finetune and store everythin
    for epoch in range(train_epochs):
        _logger.info(f"Training epoch\t{epoch}/{train_epochs}")
        adapter.train()

        batch_losses = []

        for i, (images, targets) in enumerate(tqdm(train_loader_shuffle)):
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features.to(torch.float32))
            cache_logits = get_cache_logits(affinity, cache_values, beta)

            clip_logits = 100. * image_features.to(torch.float32) @ zeroshot_weights.t()
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, targets)
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
                                        beta)
        clip_logits = 100. * val_features @ zeroshot_weights.t()
        tip_logits = clip_logits + cache_logits * alpha
        acc, f1 = get_acc_f1(tip_logits, val_labels)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            _logger.info(f"New best acc: {best_acc:.3f} \t(f1: {best_f1:.3f})")
            store_adapter(adapter, train_set.name)

    # search the best alpha and beta
    _logger.info("Loading best state dict for hp search....")
    adapter.load_state_dict(load_adapter(train_set.name))
    adapter.eval()

    best_beta, best_alpha = search_hp(cache_keys=cache_keys,
                                      cache_values=cache_values,
                                      features=val_features,
                                      labels=val_labels,
                                      clip_weights=zeroshot_weights,
                                      adapter=adapter)
    return best_alpha, best_beta


def store_adapter(model, dataset):
    adapter_path = os.path.join(Config.DATAPATH, 'tip-adapter', dataset)
    os.makedirs(adapter_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(adapter_path, 'tip_adapter.pt'))


def load_adapter(dataset):
    adapter_path = os.path.join(Config.DATAPATH, 'tip-adapter', dataset, 'tip_adapter.pt')
    return torch.load(adapter_path, map_location=Config.DEVICE)

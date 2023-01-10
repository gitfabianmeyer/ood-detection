import logging
from collections import defaultdict
import random

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
                 clip_model,
                 cache_keys):
        super().__init__()
        self.linear1 = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype)
        self.linear1.weight = nn.Parameter(cache_keys.t())


def zeroshot(clip_logits, test_labels):
    return get_acc_f1(clip_logits, test_labels)


def get_adapter_weights(dataset, model, kshots=16, train_epoch=20, alpha=1., beta=1.17, lr=0.001, eps=1e-4):
    _logger.info("Initializing everything...")
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    clip_model.eval()
    train_set = get_train_set(dataset, kshots)
    cache_keys, cache_values = get_train_features(train_set, clip_model)

    test_features, test_labels, label_features, classes = get_test_features(dataset, clip_model, clip_transform)
    _logger.info(f"Running TIP Adapter - FINETUNING")

    train_loader_shuffle = DataLoader(train_set,
                                      batch_size=256,
                                      shuffle=True,
                                      num_workers=1)
    adapter = WeightAdapter(model, cache_keys).to(device)
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

            affinity = adapter.linear1(image_features.to(torch.float32))
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

        affinity = adapter.linear1(test_features)
        cache_logits = get_cache_logits(affinity,
                                        cache_values,
                                        beta)
        clip_logits = 100. * test_features @ label_features.t()
        tip_logits = clip_logits + cache_logits * alpha
        acc, f1 = get_acc_f1(tip_logits, test_labels)
        if acc > best_acc:
            best_acc = acc
            _logger.info(f"New best acc: {acc:.3f} (f1: {f1:.3f})")
            # best_epoch = epoch
            finetuned_adapter_weights = adapter.weight # maybe return them

    return finetuned_adapter_weights


def clip_tip_adapter(dataset, kshots=16, train_epoch=20, alpha=1., beta=1.17, lr=0.001, eps=1e-4):
    _logger.info("Initializing everything...")
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    clip_model.eval()
    train_set = get_train_set(dataset, kshots)
    print(f"len trainset: {len(train_set)}. Should be: {len(train_set.classes) * kshots} (max)")

    cache_keys, cache_values = get_train_features(train_set, clip_model)

    test_features, test_labels, label_features, classes = get_test_features(dataset, clip_model, clip_transform)

    clip_logits = 100. * test_features @ label_features.t()
    zsa, f1 = zeroshot(clip_logits, test_labels)
    acc_tip_no, f1_tip_no = zeroshot_tip_no_finetuning(test_features,
                                                       cache_keys,
                                                       cache_values,
                                                       clip_logits,
                                                       test_labels,
                                                       alpha,
                                                       beta)
    acc_tip_fine, f1_tip_fine = zeroshot_tip_finetuned(train_set, clip_model,
                                                       cache_keys,
                                                       cache_values,
                                                       test_features, test_labels,
                                                       label_features, alpha,
                                                       beta, lr,
                                                       eps, train_epoch)
    results = {"zsa": zsa, "zf1": f1, "tip acc no finetuning": acc_tip_no, "tip f1 no finetuning": f1_tip_no,
               "tip acc with finetuning": acc_tip_fine, "tip f1 with finetuning": f1_tip_fine}
    return results


def get_label_dict(train_images):
    split_by_label_dict = defaultdict(list)

    # build kshot set
    for i in range(len(train_images)):
        split_by_label_dict[train_images.targets[i]].append(train_images.data[i])

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
    print(F"Truncated: {len(imgs), len(targets)}\n {targets}")
    return imgs, targets


def get_kshot_train_set(dataset, kshots):
    label_dict = get_label_dict(dataset)
    imgs, targets = get_truncated_to_min(label_dict, kshots)

    dataset.data = imgs
    dataset.targets = targets
    return dataset


def get_train_set(dataset, kshots):
    _logger.info("Creating train set")
    train_transform = get_train_transform()
    dataset = dataset(data_path=Config.DATAPATH,
                      train=True,
                      transform=train_transform)

    return get_kshot_train_set(dataset, kshots)


@torch.no_grad()
def get_train_features(train_set, model, augment_epochs=1):
    _logger.info(f'Getting train features aggregated...')

    train_loader = DataLoader(train_set,
                              batch_size=16,
                              num_workers=1,
                              shuffle=False)

    cache_values = []
    train_images_features_agg = []

    for augment_idx in range(augment_epochs):

        _logger.info(f"Augmenting features {augment_idx}/{augment_epochs}")
        train_images_features = []

        for i, (images, target) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            images_features = model.encode_image(images)
            train_images_features.append(images_features)

            if augment_idx == 0:
                target = target.to(device)
                cache_values.append(target)

        images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
        train_images_features_agg.append(images_features_cat)

    train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(0)
    train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
    train_images_features_agg = train_images_features_agg.permute(1, 0)

    cache_values = F.one_hot(torch.cat(cache_values, dim=0))
    cache_keys = train_images_features_agg.to(torch.float32)
    cache_values = cache_values.to(torch.float32)

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
def get_test_features(dataset, model, transform):
    dataset = dataset(data_path=Config.DATAPATH,
                      train=False,
                      transform=transform)
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


def get_cache_logits(new_knowledge, train_images_targets, beta):
    return ((-1) * (beta * new_knowledge.to(torch.float32))).exp() @ train_images_targets


def get_acc_f1(logits, test_labels):
    logits_topk = logits.topk(1, 1, True, True)[1].t().squeeze()
    acc = accuracy_score(test_labels.cpu().numpy(), logits_topk.cpu().numpy()) * 100
    f1 = f1_score(test_labels.cpu().numpy(), logits_topk.cpu().numpy(), average='macro') * 100
    return acc, f1


def zeroshot_tip_no_finetuning(test_features, cache_keys, cache_values, clip_logits, test_labels, alpha,
                               beta):
    _logger.info(f"Running TIP Adapter - NO FINETUNING")
    # n_images * feature_size @ (num_classes * feature_size).t() --> n_images x num_classes
    affinity = test_features @ cache_keys
    cache_logits = get_cache_logits(affinity, cache_values, beta)
    tip_logits = clip_logits + cache_logits * alpha

    acc, f1 = get_acc_f1(tip_logits, test_labels)
    return acc, f1


def zeroshot_tip_finetuned(train_set, model,
                           cache_keys, cache_values,
                           test_features, test_labels,
                           label_features, alpha,
                           beta, lr,
                           eps, train_epoch):
    _logger.info(f"Running TIP Adapter - FINETUNING")

    train_loader_shuffle = DataLoader(train_set,
                                      batch_size=256,
                                      shuffle=True,
                                      num_workers=1)
    adapter = WeightAdapter(model, cache_keys).to(device)
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

            affinity = adapter.linear1(image_features.to(torch.float32))
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

        affinity = adapter.linear1(test_features)
        cache_logits = get_cache_logits(affinity,
                                        cache_values,
                                        beta)
        clip_logits = 100. * test_features @ label_features.t()
        tip_logits = clip_logits + cache_logits * alpha
        acc, f1 = get_acc_f1(tip_logits, test_labels)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            _logger.info(f"New best acc: {acc:.3f} (f1: {f1:.3f}")
            # best_epoch = epoch
            # finetuned_adapter_weights = adapter.weight # maybe return them

    return best_acc, best_f1

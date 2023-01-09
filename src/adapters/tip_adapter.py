import logging
from collections import defaultdict
import random

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import clip
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ood_detection.classification_utils import zeroshot_classifier, accuracy, macro_f1_score
from ood_detection.config import Config

_logger = logging.getLogger()


class WeightAdapter(nn.Module):
    def __init__(self,
                 clip_model,
                 train_features,
                 cls_num,
                 shots):
        super().__init__()
        self.linear1 = nn.Linear(1024, cls_num * shots, bias=False).to(clip_model.dtype)
        self.linear1.weight = nn.Parameter(torch.load(train_features).t())


class ClipTipAdapter:
    def __init__(self, dataset, kshots, augment_epochs, lr=0.001, eps=1e-4):

        self.train_images_targets = None
        self.train_features_agg = None
        self.finetuned_adapter_weights = None
        self.train_epoch = None
        self.classes = None
        self.beta = None
        self.alpha = None
        self.label_features = None
        self.test_features = None
        self.test_labels = None
        self.dataset = dataset

        self.device = Config.DEVICE
        self.augment_epochs = augment_epochs
        self.kshots = kshots
        self.lr = lr
        self.eps = eps
        self.model, self.transform = clip.load(Config.VISION_MODEL)
        self.model.eval()
        self.train_transform = self.get_train_transform()
        self.train_set = self.get_train_set()
        self.set_test_features()

        self.get_train_features()
    def get_kshot_set(self, train_images):
        _logger.info(f"Subsampling kshot ({self.kshots}) set")
        split_by_label_dict = defaultdict(list)

        # build kshot set
        for i in range(len(train_images)):
            split_by_label_dict[train_images.targets[i]].append(train_images.data[i])
        return split_by_label_dict

    def get_train_set(self):
        _logger.info("Creating train set")
        dataset = self.dataset(data_path=Config.DATAPATH,
                               train=True,
                               transform=self.train_transform)
        imgs, targets = [], []
        for label, items in self.get_kshot_set(dataset).items():
            imgs = imgs + random.sample(items, self.kshots)
            targets = targets + [label for i in range(self.kshots)]

        dataset.data = imgs
        dataset.targets = targets
        return dataset

    def compare(self):
        results = {}
        results["zeroshot"] = self.zeroshot()
        results["TIP (no finetuning)"] = self.zeroshot_tip_no_finetuning()
        results["TIP (finetuning)"] = self.zeroshot_tip_finetuned()
        return results

    @torch.no_grad()
    def get_train_features(self):
        _logger.info(f'Getting train features aggregated...')

        train_loader = DataLoader(self.train_set,
                                  batch_size=256,
                                  num_workers=8,
                                  shuffle=False)

        train_images_targets = []
        train_images_features_agg = []

        for augment_idx in range(self.augment_epochs):

            _logger.info(f"Augmenting features {augment_idx}/{self.augment_epochs}")
            train_images_features = []

            for i, (images, target) in enumerate(tqdm(train_loader)):
                images = images.to(self.device)
                images_features = self.model.encode_image(images)
                train_images_features.append(images_features)

                if augment_idx == 0:
                    target = target.to(self.device)
                    train_images_targets.append(target)

            images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
            train_images_features_agg.append(images_features_cat)

        train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(0)
        train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
        train_images_features_agg = train_images_features_agg.permute(1, 0)
        train_images_targets = F.one_hot(torch.cat(train_images_targets, dim=0))

        self.train_features_agg = train_images_features_agg.to(torch.float32)
        self.train_images_targets = train_images_targets.to(torch.float32)

    def get_train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(size=224,
                                         scale=(0.5, 1),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    @torch.no_grad()
    def set_test_features(self):
        dataset = self.dataset(data_path=Config.DATAPATH,
                               train=False,
                               transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        test_features, test_labels = [], []

        _logger.info("Getting test features...")
        for idx, (images, targets) in enumerate(tqdm(dataloader)):
            images = images.to(self.device)
            targets = targets.to(self.device)

            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            test_features.append(image_features)
            test_labels.append(targets)
        test_features = torch.cat(test_features)
        test_labels = torch.cat(test_labels)
        self.test_features = test_features.to(torch.float32)
        self.test_labels = test_labels.to(torch.float32)
        self.label_features = zeroshot_classifier(dataset.classes, dataset.templates, self.model).to(torch.float32)
        self.classes = dataset.classes

    def get_cache_logits(self, new_knowledge):
        return ((-1) * (self.beta * new_knowledge.to(torch.float32))).exp() @ self.train_images_targets

    def get_acc_f1(self, logits):

        logits_topk = logits.topk(1, 1, True, True)[1].t().squeeze()
        acc = accuracy_score(self.test_labels, logits_topk) * 100
        f1 = f1_score(self.test_labels, logits_topk, average='macro') * 100
        _logger.info(f"ACC: {acc} \t f1: {f1}")
        return acc, f1

    def zeroshot(self):
        similarity = 100 * self.test_features.to(torch.float32) @ self.label_features.t().softmax(dim=-1)
        return self.get_acc_f1(similarity)

    def zeroshot_tip_no_finetuning(self):
        _logger.info(f"Running TIP Adapter - NO FINETUNING")
        # n_images * feature_size @ (num_classes * feature_size).t() --> n_images x num_classes
        affinity = self.test_features @ self.train_features_agg
        cache_logits = self.get_cache_logits(affinity)
        clip_logits = 100. * self.test_features @ self.label_features.t()
        tip_logits = clip_logits + cache_logits * self.alpha
        return self.get_acc_f1(tip_logits)

    def zeroshot_tip_finetuned(self):
        _logger.info(f"Running TIP Adapter - FINETUNING")

        train_loader_shuffle = DataLoader(self.train_set,
                                          batch_size=256,
                                          shuffle=True)
        adapter = WeightAdapter(self.model, self.train_features_agg, len(self.classes), self.classes).to(self.device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=self.lr, eps=self.eps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.train_epoch * len(train_loader_shuffle))

        best_acc, best_epoch = 0, 0
        losses, learning_rates, accuracies = [], [], []
        for epoch in range(self.train_epoch):
            _logger.info(f"Training epoch\t{epoch}/{self.train_epoch}")
            adapter.train()
            correct_all = 0
            n = 0
            batch_losses = []

            for i, (images, targets) in enumerate(tqdm(train_loader_shuffle)):
                images = images.to(self.device)
                targets = targets.to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                affinity = adapter.linear1(image_features)
                cache_logits = self.get_cache_logits(affinity)
                clip_logits = 100. * image_features.to(torch.float32) @ self.label_features.t()
                clip_logits = clip_logits + cache_logits * self.alpha

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

            affinity = adapter(self.test_features)
            cache_logits = self.get_cache_logits(affinity)
            clip_logits = 100. * self.test_features @ self.label_features.t()
            tip_logits = clip_logits + cache_logits * self.alpha
            acc, f1 = self.get_acc_f1(tip_logits)
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch
                self.finetuned_adapter_weights = adapter.weight

        return best_acc, best_f1

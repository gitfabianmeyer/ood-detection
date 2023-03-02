import logging

import numpy as np
import torch
import wandb
from ood_detection.config import Config
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from zeroshot.utils import get_feature_dict
from zoc.baseline import FeatureSet

from zoc.baseline import LinearClassifier

_logger = logging.getLogger(__name__)


def full_linear_classification(dset, clip_model, clip_transform, lr, epochs):
    train = get_feature_dict(dset(Config.DATAPATH,
                                  transform=clip_transform,
                                  split='train'),
                             clip_model)
    val = get_feature_dict(dset(Config.DATAPATH,
                                transform=clip_transform,
                                split='val'),
                           clip_model)

    test = None


    feature_shape = clip_model.visual.output_dim
    output_shape = len(list(train.keys()))

    train_classification_head(train,
                              val,
                              test,
                              lr,
                              epochs,
                              feature_shape,
                              output_shape,
                              True)


def train_classification_head(train: FeatureSet,
                              val: FeatureSet,
                              test: FeatureSet,
                              learning_rate,
                              train_epochs,
                              features_shape,
                              output_shape,
                              wandb_logging):
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    eval_loader = DataLoader(val, batch_size=512, shuffle=True)
    # test_loader = DataLoader(test, batch_size=521, shuffle=True)

    device = Config.DEVICE

    # init model
    classifier = LinearClassifier(features_shape, output_shape)
    classifier.train()

    optimizer = AdamW(params=classifier.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    best_val_loss = np.inf

    for epoch in tqdm(range(train_epochs)):
        epoch_dict = {}

        classifier.train()
        epoch_loss = 0.
        for image_features, targets in train_loader:
            image_features = image_features.to(device).to(torch.float32)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = classifier(image_features)
            output = criterion(predictions, targets)
            output.backward()

            loss = output.detach().item()
            epoch_loss += loss
            optimizer.step()

        epoch_dict['train loss'] = epoch_loss
        epoch_dict['train mean loss'] = np.mean(epoch_loss)

        # eval
        classifier.eval()
        epoch_val_loss = 0.
        eval_accs = []

        for eval_features, eval_targets in tqdm(eval_loader):
            eval_features = eval_features.to(torch.float32).to(device)
            eval_targets = eval_targets.to(device)

            with torch.no_grad():
                eval_preds = classifier(eval_features)
                eval_loss = criterion(eval_preds, eval_targets).detach().item()

            epoch_val_loss += eval_loss
            _, indices = torch.topk(torch.softmax(eval_preds, dim=-1), k=1)
            epoch_acc = accuracy_score(eval_targets.to('cpu').numpy(), indices.to('cpu').numpy())
            eval_accs.append(epoch_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_classifier = classifier

        epoch_dict['eval loss'] = epoch_val_loss
        epoch_dict['eval mean loss'] = np.mean(epoch_val_loss)
        epoch_dict['eval accuracy'] = np.mean(eval_accs)
        _logger.info(f"Epoch {epoch} Eval Acc: {np.mean(eval_accs)}")
        if wandb_logging:
            wandb.log(epoch_dict)

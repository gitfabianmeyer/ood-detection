import logging

import numpy as np
import torch
import wandb
from ood_detection.config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from zeroshot.utils import FeatureSet

from zoc.detectors import LinearClassifier

_logger = logging.getLogger(__name__)


def train_classification_head(train: FeatureSet,
                              val: FeatureSet,
                              learning_rate,
                              train_epochs,
                              features_shape,
                              output_shape,
                              wandb_logging):
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    eval_loader = DataLoader(val, batch_size=512, shuffle=True)
    # test_loader = DataLoader(test, batch_size=521, shuffle=True)

    device = Config.DEVICE

    # init model
    classifier = LinearClassifier(features_shape, output_shape)
    classifier.train()
    classifier.to(device)

    optimizer = AdamW(params=classifier.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    best_val_acc = 0.

    for epoch in range(1, train_epochs + 1):
        epoch_dict = {}

        classifier.train()
        epoch_loss = 0.
        for batch_idx, (image_features, targets) in enumerate(train_loader):
            image_features = image_features.to(torch.float32).to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = classifier(image_features)
            output = criterion(predictions, targets)
            output.backward()

            loss = output.detach().item()
            epoch_loss += loss
            optimizer.step()

        epoch_dict['train loss'] = epoch_loss
        epoch_dict['train mean loss'] = epoch_loss / len(train_loader)

        # eval
        classifier.eval()
        epoch_val_loss = 0.
        eval_accs = []

        for eval_features, eval_targets in eval_loader:
            eval_features = eval_features.to(torch.float32).to(device)
            eval_targets = eval_targets.to(device)

            with torch.no_grad():
                eval_preds = classifier(eval_features)
                eval_loss = criterion(eval_preds, eval_targets).detach().item()

            epoch_val_loss += eval_loss
            _, indices = torch.topk(torch.softmax(eval_preds, dim=-1), k=1)
            epoch_acc = accuracy_score(eval_targets.to('cpu').numpy(), indices.to('cpu').numpy())
            eval_accs.append(epoch_acc)

        epoch_acc = np.mean(eval_accs)
        if epoch_acc > best_val_acc:
            _logger.info(f"New best epoch {epoch}\t Eval Acc: {np.mean(eval_accs)}\t Loss: {np.mean(epoch_val_loss)}")
            best_val_acc = epoch_acc
            best_classifier = classifier
            best_epoch = epoch
            best_loss = epoch_val_loss

        epoch_dict['eval loss'] = epoch_val_loss
        epoch_dict['eval mean loss'] = epoch_val_loss / len(eval_loader)
        epoch_dict['eval accuracy'] = epoch_acc
        if wandb_logging:
            wandb.log(epoch_dict)

    final = {'best val acc': best_val_acc,
             'best epoch': best_epoch,
             'best loss': best_loss}

    wandb.log(final)
    return best_val_acc, best_classifier


@torch.no_grad()
def get_test_accuracy_from_dset(test, classifier):
    device = Config.DEVICE
    classifier.eval()

    test_loader = DataLoader(test, batch_size=512)
    eval_accs = []
    for eval_features, eval_targets in test_loader:
        eval_features = eval_features.to(torch.float32).to(device)
        eval_targets = eval_targets.to(device)

        with torch.no_grad():
            eval_preds = classifier(eval_features)

        _, indices = torch.topk(torch.softmax(eval_preds, dim=-1), k=1)
        epoch_acc = accuracy_score(eval_targets.to('cpu').numpy(), indices.to('cpu').numpy())
        eval_accs.append(epoch_acc)

    return np.mean(eval_accs)

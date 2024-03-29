import logging

import torch

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def shape_printer(tensor, name):
    _logger.info(f"Shape of {name}: {tensor.shape}")


def id_ood_printer(id_classes, ood_classes):
    _logger.debug(f"id Classes: {id_classes[:2]}... OOD classes: {ood_classes[:2]}...")


def dataset_name_printer(name):
    blank_line = "-" * 30
    _logger.info(f"\n\n{blank_line}{name}{blank_line}\n")


def distance_name_printer(name):
    _logger.info(f"\t\t\tStarting {name.upper()}...")


def mean_std_printer(mean, std, runs):
    _logger.info(f"Runs: {runs}\t\tMean: {mean: .3f}\t\t std: {std: .3f}")


def accuracy_printer(accuracy):
    _logger.info(f"Zero Shot Accuracy: {accuracy: .3f}")


def debug_scores(ten, name):
    std, mean = torch.std_mean(ten)
    _logger.info(f"{name}: mean: {mean: .4f}, std: {std: .5f}")
    shape_printer(tensor=ten, name=name)
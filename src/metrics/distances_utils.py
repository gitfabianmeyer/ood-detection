import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def shape_printer(name, tensor):
    # _logger.info(f"Shape of {name}: {tensor.shape}")
    pass


def id_ood_printer(id_classes, ood_classes):
    _logger.debug(f"id Classes: {id_classes[:2]}... OOD classes: {ood_classes[:2]}...")


def dataset_name_printer(name):
    blank_line = "-" * 30
    _logger.info(f"\n\n{blank_line}{name}{blank_line}\n")


def distance_name_printer(name):
    _logger.info(f"\n\tStarting {name.upper()}...")


def mean_std_printer(mean, std, runs):
    _logger.info(f"Runs: {runs}\t\tMEAN: {mean}\t\t STD: {std}")


def accuracy_printer(accuracy):
    _logger.info(f"Zero Shot Accuracy: {accuracy}")

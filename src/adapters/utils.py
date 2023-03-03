import logging

from adapters.tip_adapter import get_acc_f1


_logger = logging.getLogger(__name__)


def zeroshot(clip_logits, test_labels):
    return get_acc_f1(clip_logits, test_labels)

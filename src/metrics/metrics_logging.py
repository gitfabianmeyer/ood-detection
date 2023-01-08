import logging

import wandb
from datetime import datetime
from ood_detection.config import Config

_logger = logging.getLogger(__name__)


def wandb_log(metrics_dict, experiment="distances"):
    if "corruption" in metrics_dict.keys():
        name = "-".join([metrics_dict["dataset"], metrics_dict["corruption"], datetime.today().strftime('%Y/%m/%d')])
    else:
        name = "-".join([metrics_dict["dataset"], datetime.today().strftime('%Y/%m/%d')])
        metrics_dict["corruption"] = "No Corruption"

    if experiment == 'zsoodd_corruptions':
        _logger.info(f'logging {experiment}')
        run = wandb.init(project="thesis-zsoodd-corruptions",
                         entity="wandbefab",
                         name=name,
                         tags=[
                             'zeroshot',
                             'zsoodd',
                             'oodd',
                             metrics_dict['corruption'],
                             metrics_dict['dataset'],
                             metrics_dict['model'],
                         ])

        wandb.config = {
            "batch_size": 512,
            "clip_model": Config.VISION_MODEL,
            "device": Config.DEVICE,
        }

    elif experiment == "distances":
        _logger.info(f'logging {experiment}')

        run = wandb.init(project="thesis-datasets",
                         entity="wandbefab",
                         name=name,
                         tags=['zeroshot',
                               'mmd',
                               'clp',
                               'zsa',
                               metrics_dict["corruption"],
                               metrics_dict["dataset"],
                               metrics_dict["model"]],
                         )
    elif experiment == 'zsoodd':
        _logger.info(f'logging {experiment}')

        run = wandb.init(project="thesis-zsoodd",
                         entity="wandbefab",
                         name=name,
                         tags=[
                             'zeroshot',
                             'zsoodd',
                             'oodd',
                             metrics_dict['corruption'],
                             metrics_dict['dataset'],
                             metrics_dict['model'],
                         ])

        wandb.config = {
            "batch_size": 512,
            "clip_model": Config.VISION_MODEL,
            "device": Config.DEVICE,
        }

    else:
        _logger.error(f"{experiment} could not be found as logging metric.")
        raise ValueError

    # remove strings
    try:
        metrics_dict.pop("dataset")
        metrics_dict.pop("model")
        metrics_dict.pop("corruption")
    except KeyError:
        pass
    wandb.log(
        metrics_dict
    )
    return run

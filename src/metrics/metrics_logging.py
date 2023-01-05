import wandb
from datetime import datetime
from ood_detection.config import Config


def wandb_log(metrics_dict, experiment="distances"):
    if "corruption" in metrics_dict.keys():
        name = "-".join([metrics_dict["dataset"], metrics_dict["corruption"], datetime.today().strftime('%Y/%m/%d')])
    else:
        name = "-".join([metrics_dict["dataset"], datetime.today().strftime('%Y/%m/%d')])
        metrics_dict["corruption"] = "No Corruption"
    if experiment == "distances":
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
        return

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

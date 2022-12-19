import wandb
from datetime import datetime
from ood_detection.config import Config


def wandb_log(metrics_dict, experiment="distances"):
    name = "-".join([metrics_dict["dataset"], datetime.today().strftime('%Y/%m/%d')])
    if "corruption" in metrics_dict.keys():
        name = "-".join([metrics_dict["dataset"], metrics_dict["corruption"], datetime.today().strftime('%Y/%m/%d')])
    if experiment == "distances":
        run = wandb.init(project="thesis-datasets",
                         entity="wandbefab",
                         name=name,
                         tags=['zeroshot',
                               'mmd',
                               'clp',
                               'zsa',
                               metrics_dict["corruption"],
                               metrics_dict["name"],
                               metrics_dict["model"]],
                         )

        wandb.config = {
            "batch_size": 512,
            "clip_model": Config.VISION_MODEL,
            "device": Config.DEVICE,
        }

    else:
        return

    # remove strings
    metrics_dict.pop("corruption")
    metrics_dict.pop("name")
    metrics_dict.pop("model")
    wandb.log(
        metrics_dict
    )
    return run

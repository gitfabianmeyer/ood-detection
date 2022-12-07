import wandb
from datetime import datetime
from ood_detection.config import Config


def wandb_log(metrics_dict, experiment="distances"):
    if experiment == "distances":
        wandb.init(project="thesis-datasets",
                   entity="wandbefab",
                   name="-".join([metrics_dict["name"], datetime.today().strftime('%Y/%m/%d')]),
                   tags=['zeroshot', 'mmd', 'clp', 'zsa'],
                   )
        wandb.config = {
            "batch_size": 512,
            "clip_model": Config.VISION_MODEL,
            "device": Config.DEVICE,
        }

    else:
        return
    wandb.log(
        metrics_dict
    )

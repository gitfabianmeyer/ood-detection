import wandb
from ood_detection.config import Config


def wandb_log(metrics_dict):
    wandb.init(project="thesis-datasets", entity="wandbefab")
    wandb.config = {
        "batch_size": 512,
        "clip_model": Config.VISION_MODEL,
        "device": Config.DEVICE,
    }

    wandb.log(
        metrics_dict
    )

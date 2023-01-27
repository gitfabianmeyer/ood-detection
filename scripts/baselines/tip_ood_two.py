import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import clip
from ood_detection.config import Config
from adapters.ood import tip_ood_detector

import logging

import wandb
from datasets.config import HalfTwoDict

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

run_clearml = False
runs = 10
kshots = 16
train_epochs = 1
augment_epochs = 10


def main():
    failed = []
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    device = Config.DEVICE

    for dname, dset in HalfTwoDict.items():

        _logger.info(f"\t\tStarting {dname} run...")
        run = wandb.init(project=f"thesis-tip-ood-test-50-runs",
                         entity="wandbefab",
                         name=dname)
        try:
            results = tip_ood_detector(dset,
                                       clip_model,
                                       clip_transform,
                                       device,
                                       Config.ID_SPLIT,
                                       runs,
                                       kshots,
                                       augment_epochs)
            print(results)
        except Exception as e:
            failed.append(dname)
            raise e

        wandb.log(results)
        run.finish()

    print(f"Failed: {failed}")


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

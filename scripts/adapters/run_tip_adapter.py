import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datetime import datetime
import logging

import wandb
from datasets.config import DATASETS_DICT

from adapters.tip_adapter import clip_tip_adapter

# import for clearml
import clip
import torch
import torchvision
import numpy as np
import sklearn
import tqdm

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
run_clearml = False


def main():

    kshots = 16
    failed = []
    for dname, dset in DATASETS_DICT.items():

        if dname != 'gtsrb':
            _logger.info(f"Jumping over {dname}")
            continue

        _logger.info(f"\t\tStarting {dname} run...")

        try:
            results = clip_tip_adapter(dataset=dset,
                                       kshots=16,
                                       train_epoch=20,
                                       alpha=1.,
                                       beta=1.17,
                                       lr=0.001,
                                       eps=1e-4)
            print(results)
        except Exception as e:
            failed.append(dname)
        run = wandb.init(project=f"thesis-tip-adapters-{kshots}_shots",
                         entity="wandbefab",
                         name=dname)
        run.log(results)
        run.finish()

    print(f"Failed: {failed}")


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

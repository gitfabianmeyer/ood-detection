from datetime import datetime
import logging
import os

import wandb
from datasets.config import DATASETS_DICT

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    for dname, dset in DATASETS_DICT:
        _logger.info(f"Starting {dname} run...")

        name = "-".join([dname, 'adapter', datetime.today().strftime('%Y/%m/%d')])
        run = wandb.init(project="thesis-zsoodd-corruptions",
                         entity="wandbefab",
                         name=name)
        try:
            results = clip_tip_adapter(dataset=dset)
            run.log(results)
        except:
            pass
        run.finish()


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

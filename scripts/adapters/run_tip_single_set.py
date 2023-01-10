import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    failed = []
    for dname, dset in DATASETS_DICT.items():

        if dname in ['caltech101', 'caltech cub', 'dtd', 'fashion mnist', 'flowers102', 'imagenet', 'lsun', 'mnist']:

            _logger.info(f"Starting {dname} run...")


            try:
                results = clip_tip_adapter(dataset=dset)
                print(results)
            except Exception as e:
                failed.append(dname)
                raise e
                break
            name = "-".join([dname, 'adapter', datetime.today().strftime('%Y/%m/%d')])
            run = wandb.init(project="thesis-tip-adapters",
                             entity="wandbefab",
                             name=name)
            run.log(results)
            run.finish()
        else:
            print(f"Jumping over {dname}, already exists")
    print(f"Failed: {failed}")


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

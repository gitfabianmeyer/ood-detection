import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging

import wandb
from datasets.config import DATASETS_DICT

from adapters.tip_adapter import full_clip_tip_classification

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
run_clearml = False

kshots = 16
train_epochs = 20
lr = 0.001
eps = 1e-4
augment_epochs = 10


def main():
    failed = []

    for dname, dset in DATASETS_DICT.items():
        _logger.info(f"\t\tStarting {dname} run...")
        try:
            results = full_clip_tip_classification(dataset=dset,
                                                   kshots=kshots,
                                                   train_epochs=train_epochs,
                                                   lr=lr,
                                                   eps=eps,
                                                   augment_epochs=augment_epochs)
            print(results)
        except Exception as e:

            failed.append(dname)
            raise e
        run = wandb.init(project=f"thesis-tip-adapters-{kshots}_shots-test",
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

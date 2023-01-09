from clearml import Task
from datasets import config

from adapters.tip_adapter import ClipTipAdapter

run_clearml = True


def main():
    dataset = config.DATASETS_DICT["cifar10"]

    tip_adapter = ClipTipAdapter(dataset=dataset,
                                 kshots=16,
                                 augment_epochs=1,
                                 lr=0.001,
                                 eps=1e-4)

    result = tip_adapter.compare()
    return result


if __name__ == '__main__':
    if run_clearml:
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')

    main()

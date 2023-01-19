import numpy as np
import wandb
from datasets.config import DATASETS_DICT
from ood_detection.config import Config


def calc_openness(classes, split):
    total_classes = len(classes)
    id_classes = int(total_classes * split)
    test_id_classes = id_classes

    return (1 - np.sqrt((2 * id_classes) / (test_id_classes + total_classes))) * 100


def main():
    for dname, dset in DATASETS_DICT.items():
        run = wandb.init(project="thesis-openness",
                         entity="wandbefab",
                         name=dname)
        dataset = dset(Config.DATAPATH,
                       transform=None,
                       split='test')

        openess = calc_openness(dataset.classes, .4)
        wandb.log({'openness': openess,
                   'classes': len(dataset.classes)})
        run.finish()
        print(dname)
        print(openess)
        print("______________")

    run = wandb.init(project="thesis-openness",
                     entity="wandbefab",
                     name="Base")
    openess_base = calc_openness([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], .4)
    wandb.log({'openness': openess_base,
               'classes': 10})
    run.finish()


if __name__ == '__main__':
    main()

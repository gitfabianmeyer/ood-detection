import clip
import wandb
from datasets import config
from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config


def main():
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    for dname, dset in config.DATASETS_DICT.items():

        if dname == 'lsun':
            lsun = True
        else:
            lsun = False

        run = wandb.init(project="thesis-dataset_distances_val_no_corruptions_40",
                         entity="wandbefab",
                         name=dname,
                         tags=["distances",
                               'statistics'])
        dataset = dset(Config.DATAPATH,
                       split='val',
                       transform=clip_transform)
        metrics = get_distances_for_dataset(dataset, clip_model, splits=5, id_split=.4, corruption=None, severity=None,
                                            lsun=lsun)

        wandb.log(metrics)
        run.finish()


if __name__ == '__main__':
    main()

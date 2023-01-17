import clip
import wandb
from datasets import config
from metrics.distances import get_distances_for_dataset
from ood_detection.config import Config

clip_model, clip_transform = clip.load(Config.DATAPATH)
for dset, dname in config.DATASETS_DICT:

    if dname == 'lsun':
        lsun = True
    else:
        lsun = False

    run = wandb.init(project="thesis-dataset_distances_val_no_corruptions_40/60",
                     entity="wandbefab",
                     name=dname,
                     tags=["distances",
                           'statistics'])
    dataset = dset(Config.DATAPATH,
                   split='val',
                   transform=clip_transform)
    metrics = get_distances_for_dataset(dset, clip_model, splits=5, id_split=.4, corruption=None, severity=None,
                                        lsun=lsun)

    wandb.log(metrics)
    run.finish()

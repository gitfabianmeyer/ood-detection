import logging
import random

import clip
import numpy as np
import wandb
from datasets.config import DATASETS_DICT
from datasets.zoc_loader import IsolatedClasses
from metrics.distances import Distancer, MaximumMeanDiscrepancy, ConfusionLogProbability
from ood_detection.config import Config

_logger = logging.getLogger(__name__)
# run this script 2 times: 1. with imagenet templates, 2 with custom templates

for dname, dset in DATASETS_DICT.items():

    run = wandb.init(project="thesis-clp-temperature-10_classes_100runs",
                     entity="wandbefab",
                     name=dname,
                     tags=['distance',
                           'metrics'])

    if dname == 'lsun':
        lsun = True
    else:
        lsun = False

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    dataset = dset(Config.DATAPATH,
                   split='val',
                   transform=clip_transform)

    loaders = IsolatedClasses(dataset, batch_size=512, lsun=False)

    splits = 100  # run each exp 10 times
    id_split = Config.ID_SPLIT

    _logger.info("Initializing distancer")
    distancer = Distancer(isolated_classes=loaders,
                          clip_model=clip_model,
                          splits=splits,
                          id_split=id_split)

    for temperature in np.logspace(-7.158429362604483, 6.643856189774724, num=50,
                                   base=2.0):

        temperature_results = []

        # subsample 10
        for i in range(splits):
            # subsample 10
            subsample = random.sample(distancer.feature_dict.keys(), 10)
            subsampled_feature_dict = {key: distancer.feature_dict[key] for key in subsample}

            clp = ConfusionLogProbability(subsampled_feature_dict, clip_model)
            temperature_results.append(clp.get_distance())

        wandb.log({"temperature": temperature,
                   'clp': np.mean(temperature_results),
                   'std': np.std(temperature_results)})

    run.finish()

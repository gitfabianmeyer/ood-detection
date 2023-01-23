import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging
import clip
import numpy as np
import torch
import wandb
from datasets.corruptions import get_corruption_transform, THESIS_CORRUPTIONS, store_corruptions_feature_dict
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config

from datasets.config import DATASETS_DICT
from metrics.distances import run_full_distances, MaximumMeanDiscrepancy, ConfusionLogProbability, Distancer, \
    ZeroShotAccuracy

_logger = logging.getLogger(__name__)

clip_model, clip_transform = clip.load(Config.VISION_MODEL)


def main():
    for dname, dset in DATASETS_DICT.items():
        if dname not in ['gtsrb']:
            continue
        for cname, ccorr in THESIS_CORRUPTIONS.items():
            if cname != 'Glass Blur':
                continue
            run = wandb.init(project="thesis-full_distances-selected-sets",
                             entity="wandbefab",
                             name="_".join([dname, cname]),
                             tags=['distance',
                                   'metrics'])
            for severity in [1, 3, 5]:

                _logger.info(f"Running {dname} with {cname} and severity {severity}")

                corruption_transform = get_corruption_transform(clip_transform, ccorr, severity)

                dataset = dset(Config.DATAPATH,
                               split='val',
                               transform=corruption_transform)

                loaders = IsolatedClasses(dataset, batch_size=512, lsun=False)

                splits = 10  # run each exp 10 times
                id_split = Config.ID_SPLIT

                _logger.info("Initializing distancer")

                distancer = Distancer(isolated_classes=loaders,
                                      clip_model=clip_model,
                                      splits=splits,
                                      id_split=id_split)

                store_corruptions_feature_dict(distancer.feature_dict, cname, dname, severity)
                store_corruptions_feature_dict(distancer.feature_dict, cname, dname, severity)

                clp = ConfusionLogProbability(distancer.feature_dict, clip_model)
                mmd = MaximumMeanDiscrepancy(distancer.feature_dict)
                zsa = ZeroShotAccuracy(distancer.feature_dict,
                                       clip_model,
                                       distancer.targets)

                clp_results, mmd_results = [], []
                zsa_result = zsa.get_distance()["zsa"]

                for i in range(splits):
                    clp_results.append(clp.get_distance())
                    mmd_results.append(mmd.get_distance())

                # zsa is independent from splits

                wandb.log({"zsa": zsa_result,
                           'clp': np.mean(clp_results),
                           'clp_std': np.std(clp_results),
                           'mmd': np.mean(mmd_results),
                           'mmd_std': np.std(mmd_results),
                           'severity': severity})

            run.finish()


if __name__ == '__main__':
    main()

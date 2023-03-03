import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import clip
import numpy as np
import torch
import wandb
from datasets.corruptions import get_corruption_transform, THESIS_CORRUPTIONS, store_corruptions_feature_dict
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config

from datasets.config import DATASETS_DICT, CorruptionSets
from metrics.distances import run_full_distances, MaximumMeanDiscrepancy, ConfusionLogProbability, Distancer, \
    ZeroShotAccuracy

_logger = logging.getLogger(__name__)

clip_model, clip_transform = clip.load(Config.VISION_MODEL)


def main():
    for dname, dset in CorruptionSets.items():
        if dname not in ['gtsrb']:
            continue
        for cname, ccorr in THESIS_CORRUPTIONS.items():
            if cname == 'Glass Blur':
                continue
            run = wandb.init(project="thesis-corruption-distances",
                             entity="wandbefab",
                             name="_".join([dname, cname]),
                             tags=['distance',
                                   'metrics'])
            for severity in [1, 3, 5]:

                _logger.info(f"Running {dname} with {cname} and severity {severity}")

                corruption_transform = get_corruption_transform(clip_transform, ccorr, severity)

                dataset = dset(Config.DATAPATH,
                               split='train',
                               transform=corruption_transform)

                loaders = IsolatedClasses(dataset, batch_size=512, lsun=False)

                runs = 10  # run each exp 10 times
                id_split = Config.ID_SPLIT

                _logger.info("Initializing distancer")

                distancer = Distancer(isolated_classes=loaders,
                                      clip_model=clip_model,
                                      splits=runs,
                                      id_split=id_split)

                clp = ConfusionLogProbability(distancer.feature_dict, clip_model)
                mmd = MaximumMeanDiscrepancy(distancer.feature_dict)
                zsa = ZeroShotAccuracy(distancer.feature_dict,
                                       clip_model,
                                       distancer.targets)



                # zsa doesn't change!
                zsa_result = zsa.get_distance()["zsa"]
                clp_results, mmd_results = [], []
                for i in range(runs):
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

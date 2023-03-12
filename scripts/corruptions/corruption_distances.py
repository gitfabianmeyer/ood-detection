import torch


def run_all(args):
    import logging
    import clip
    import numpy as np
    import wandb
    from datasets.corruptions import get_corruption_transform, THESIS_CORRUPTIONS
    from datasets.zoc_loader import IsolatedClasses
    from ood_detection.config import Config
    from datasets.config import CorruptionSets
    from metrics.distances import MaximumMeanDiscrepancy, ConfusionLogProbability, Distancer, \
        ZeroShotAccuracy

    _logger = logging.getLogger(__name__)

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)

    for dname, dset in CorruptionSets.items():
        for cname, ccorr in THESIS_CORRUPTIONS.items():

            run = wandb.init(project="thesis-corruption-distances-final",
                             entity="wandbefab",
                             name="_".join([dname, cname]),
                             tags=['distance',
                                   'metrics'])
            for severity in [1, 3, 5]:

                _logger.info(f"Running {dname} with {cname} and severity {severity}")

                corruption_transform = get_corruption_transform(clip_transform, ccorr, severity)

                dataset = dset(Config.DATAPATH,
                               split='test',
                               transform=corruption_transform)

                runs = args.runs  # run each exp 10 times

                _logger.info("Initializing distancer")
                from zeroshot.utils import FeatureDict
                feature_dict = FeatureDict(dataset, clip_model)

                clp = ConfusionLogProbability(feature_dict, clip_model)
                mmd = MaximumMeanDiscrepancy(feature_dict)
                zsa = ZeroShotAccuracy(feature_dict,
                                       clip_model,
                                       torch.Tensor(dataset.targets))

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


def main(args):
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_all(args)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--runs", type=int, default=10)

    args = parser.parse_args()
    main(args)

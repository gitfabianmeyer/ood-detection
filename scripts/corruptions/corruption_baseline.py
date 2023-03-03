import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from datasets.corruptions import THESIS_CORRUPTIONS, store_corruptions_feature_dict, load_corruptions_feature_dict, \
    get_corruption_transform

from datasets.config import CorruptionSets
from zoc.utils import get_feature_dict_from_isolated_classes

import argparse
import logging
import clip
import wandb
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config

_logger = logging.getLogger(__name__)
create_features = False


def unpack_mean_metrics(metrics):
    new = {'auc-max': np.mean(metrics['auc-max']), 'auc-max-std': np.std(metrics['auc-max']),
           'f1': np.mean(metrics['f1_mean']), 'f1-std': np.std(metrics['f1_mean']),
           'auc-mean': np.mean(metrics['auc-mean']), 'auc-mean-std': np.std(metrics['auc-mean']),
           'acc': np.mean(metrics['acc_mean']), 'acc-std': np.std(metrics['acc_mean']),
           'auc-sum': np.mean(metrics['auc-sum']), 'auc-sum-std': np.std(metrics['auc-sum'])}
    return new


def run_all(args):
    # for each dataset
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    for dname, dset in CorruptionSets.items():
        for cname, ccorr in THESIS_CORRUPTIONS.items():
            run = wandb.init(project="thesis-corruptions-zoc-all_classes-baseline",
                             entity="wandbefab",
                             name="_".join([dname, cname]),
                             tags=['distance',
                                   'metrics'])
            for severity in [1, 3, 5]:
                _logger.info(f"---------------- Running {dname} with {cname} and severity {severity} ---------------")
                transform = get_corruption_transform(clip_transform, ccorr, severity)
                dataset = dset(data_path=Config.DATAPATH,
                               split='test',
                               transform=transform)

                isolated_classes = IsolatedClasses(dataset,
                                                   batch_size=512)
                if create_features:
                    _logger.info('Creating corruptions set')
                    feature_dict = get_feature_dict_from_isolated_classes(isolated_classes, clip_model)
                    store_corruptions_feature_dict(feature_dict, cname, dname + '-test', severity)

                else:
                    _logger.info("Loading feature dict...")
                    feature_dict = load_corruptions_feature_dict(isolated_classes.classes, cname, dname + '-test',
                                                                 severity)
                metrics = baseline_detector_no_temperature_featuredict(feature_dict,
                                                                       dset,
                                                                       clip_model,
                                                                       clip_transform,
                                                                       Config.DEVICE,
                                                                       id_classes_split=Config.ID_SPLIT,
                                                                       runs=args.runs_ood)


                to_log = unpack_mean_metrics(metrics)
                to_log["severity"] = severity
                wandb.log(to_log)
            run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_ood', type=int, default=5)

    args = parser.parse_args()
    run_all(args)

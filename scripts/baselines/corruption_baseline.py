import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from datasets.corruptions import THESIS_CORRUPTIONS, store_corruptions_feature_dict, load_corruptions_feature_dict

from datasets.config import CorruptionSets
from zoc.baseline import baseline_detector_no_temperature_featuredict
from zoc.utils import get_feature_dict_from_isolated_classes

import argparse
import logging
import clip
import wandb
from datasets.zoc_loader import IsolatedClasses
from ood_detection.config import Config

_logger = logging.getLogger(__name__)
splits = [(.4, .6), ]
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
            run = wandb.init(project="thesis-zoc-selected_corruption-selected_sets_all-classes-baseline",
                             entity="wandbefab",
                             name="_".join([dname, cname]),
                             tags=['distance',
                                   'metrics'])
            for severity in [1, 3, 5]:
                _logger.info(f"---------------- Running {dname} with {cname} and severity {severity} ---------------")

                if dname == 'lsun':
                    lsun = True

                else:
                    lsun = False

                dataset = dset(data_path=Config.DATAPATH,
                               split='test',
                               transform=clip_transform)

                # shorted_classes = random.sample(dataset.classes, 10)
                # dataset.classes = shorted_classes

                isolated_classes = IsolatedClasses(dataset,
                                                   batch_size=512,
                                                   lsun=lsun)
                if create_features:
                    _logger.info('Creating corruptions set')
                    feature_dict = get_feature_dict_from_isolated_classes(isolated_classes, clip_model)
                    store_corruptions_feature_dict(feature_dict, cname, dname + '-test', severity)

                else:
                    _logger.info("Loading feature dict...")
                    feature_dict = load_corruptions_feature_dict(isolated_classes.classes, cname, dname + '-test',
                                                                 severity)
                    # feature_dict = load_corruptions_feature_dict(isolated_classes.classes, cname, dname, severity)

                for split in splits:
                    # perform zsoodd
                    metrics = baseline_detector_no_temperature_featuredict(feature_dict,
                                                                           dset,
                                                                           clip_model,
                                                                           clip_transform,
                                                                           Config.DEVICE,
                                                                           id_classes_split=split[0],
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

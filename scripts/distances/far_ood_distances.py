import logging

_logger = logging.getLogger(__name__)


def run_all(args):
    import clip
    import numpy as np
    from datasets.config import DATASETS_DICT
    from tqdm import tqdm
    import wandb
    from metrics.distances import get_far_mmd, get_far_clp
    from zeroshot.utils import FeatureDict
    from ood_detection.config import Config
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)

    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")
    failed = []

    for id_dname in datasets:
        id_dset = DATASETS_DICT[id_dname]

        _logger.info(f"Running all datasets distances for {id_dname}")
        dataset = id_dset(Config.DATAPATH,
                          split='train',
                          transform=clip_transform)
        dataset_featuredict = FeatureDict(dataset, clip_model)

        for ood_dname, ood_dset in tqdm(DATASETS_DICT.items()):
            if id_dname == ood_dname:
                continue
            run = wandb.init(project=f"thesis-far-distances",
                             entity="wandbefab",
                             name=id_dname + '-' + ood_dname)

            _logger.info(F"Running id: {id_dname} vs ood: {ood_dname}")
            ood_featuredict = FeatureDict(ood_dset(Config.DATAPATH,
                                                   split='train',
                                                   transform=clip_transform),
                                          clip_model)
            ex = None
            try:
                mmd = get_far_mmd(dataset_featuredict, ood_featuredict)
            except Exception as e:
                ex = "MMD"
                mmd = "FAILED"
            try:
                clp = get_far_clp(dataset_featuredict, ood_featuredict, clip_model, 1)
            except:
                ex = "CLP"
                clp = 'FAILED'

            wandb.log({'mmd': mmd, 'clp': clp})

            if ex:
                failed.append((id_dname, ood_dname, str(ex)))
            run.finish()
    print(failed)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--split", type=int)
    parser.add_argument('--split', type=int)
    parser.add_argument('--max_split', type=int)
    args = parser.parse_args()

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_all(args)


if __name__ == '__main__':
    main()

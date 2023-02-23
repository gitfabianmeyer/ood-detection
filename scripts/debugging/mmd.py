import logging

_logger = logging.getLogger(__name__)


def run_all(args):
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    import clip
    from datasets.config import HalfOneDict, HalfTwoDict, DATASETS_DICT
    from tqdm import tqdm
    import wandb
    from metrics.distances import get_far_mmd, get_far_clp
    from zeroshot.utils import FeatureDict
    from ood_detection.config import Config
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)

    if args.split == 1:
        datasets = HalfOneDict
    elif args.split == 2:
        datasets = HalfTwoDict
    else:
        datasets = DATASETS_DICT
    failed = []
    for id_dname, id_dset in datasets.items():
        if id_dname != args.id_name:
            continue
        _logger.info(f"Running all datasets distances for {id_dname}")
        dataset = id_dset(Config.DATAPATH,
                          split='train',
                          transform=clip_transform)
        dataset_featuredict = FeatureDict(dataset, clip_model)

        for ood_dname, ood_dset in tqdm(DATASETS_DICT.items()):
            if id_dname == ood_dname or ood_dname != args.ood_name:
                continue
            # run = wandb.init(project=f"thesis-{id_dname}-far-distances",
            #                  entity="wandbefab",
            #                  name=ood_dname)

            _logger.info(F"Running id: {id_dname} vs ood: {ood_dname}")
            ood_featuredict = FeatureDict(ood_dset(Config.DATAPATH,
                                                   split='train',
                                                   transform=clip_transform),
                                          clip_model)
            ex = None
            try:
                mmd = get_far_mmd(dataset_featuredict, ood_featuredict)
            except Exception as e:
                print(e)
                ex = "MMD"
                mmd = "FAILED"
            try:
                clp = get_far_clp(dataset_featuredict, ood_featuredict, clip_model, 1)
            except Exception as e:
                print(e)
                ex = "CLP"
                clp = 'FAILED'


            # wandb.log({'mmd': mmd, 'clp': clp})
            print({'mmd': mmd, 'clp': clp})
            if ex:
                failed.append((id_dname, ood_dname, str(ex)))
            # run.finish()
    print(failed)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--split", type=int)
    parser.add_argument("--id_name", type=str)
    parser.add_argument("--ood_name", type=str)
    args = parser.parse_args()

    run_all(args)


if __name__ == '__main__':
    main()

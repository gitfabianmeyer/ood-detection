import logging

_logger = logging.getLogger(__name__)


def main(args):
    from adapters.tip_adapter import full_clip_tip_classification
    from datasets.config import DATASETS_DICT
    import wandb

    failed = []

    if args.split == 0:
        datasets = DATASETS_DICT.keys()
    else:
        import numpy as np
        datasets_splits = np.array_split(list(DATASETS_DICT.keys()), args.max_split)
        datasets = datasets_splits[args.split - 1]
        _logger.info(f"Current split: {args.split}: {datasets}")

    for dname in datasets:
        dset = DATASETS_DICT[dname]
        _logger.info(f"\t\tStarting {dname} run...")
        run = wandb.init(project=f"thesis-tip-adapters-{args.kshots}_shots-temp_{args.temp}",
                         entity="wandbefab",
                         name=dname,
                         config=args.__dict__)
        try:
            for i in range(args.runs):
                results = full_clip_tip_classification(dataset=dset,
                                                       kshots=args.kshots,
                                                       train_epochs=args.train_epochs,
                                                       lr=args.lr,
                                                       eps=args.eps,
                                                       augment_epochs=args.augment_epochs,
                                                       temperature=args.temp)
                run.log(results)

        except Exception as e:

            failed.append(dname)
            raise e

        run.finish()

    print(f"Failed: {failed}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--kshots", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--augment_epochs", type=int, default=10)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--max_split", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.01)
    args = parser.parse_args()
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)

def run_single(args):
    import numpy as np
    import wandb
    from datasets.config import DATASETS_DICT
    from adapters.tip_adapter import full_clip_tip_classification


    if args.dname != 'all':
        dname = args.dname
        dset = DATASETS_DICT[dname]

        datasets ={dname:dset}
    else:
        datasets = DATASETS_DICT

    for dname, dset in datasets.items():

        temperatures = np.logspace(np.log2(0.001), np.log2(100), num=args.temperatures, base=2.0)
        run = wandb.init(project=f"thesis-ablation-tip-temperatures",
                         entity="wandbefab",
                         name=dname,
                         config=vars(args))

        for temp in temperatures:
            results = full_clip_tip_classification(dataset=dset,
                                                   kshots=args.shots,
                                                   train_epochs=args.train_epochs,
                                                   init_alpha=1.,
                                                   init_beta=1.,
                                                   lr=args.lr,
                                                   eps=args.eps,
                                                   augment_epochs=args.augment_epochs,
                                                   temperature=temp)
            wandb.log(results)
        run.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--temperatures", type=int, default=10)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--augment-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--dname", type=str, default='all')

    args = parser.parse_args()
    print(vars(args))
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_single(args)


if __name__ == '__main__':
    main()

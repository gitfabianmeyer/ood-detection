def run_single(args):
    import clip
    import numpy as np
    import wandb
    from datasets.config import DATASETS_DICT
    from ood_detection.config import Config
    from zoc.ablation import zoc_temp_ablation

    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    dset = DATASETS_DICT[args.dname]
    device = Config.DEVICE

    temperatures = np.logspace(np.log2(0.01), np.log2(100), num=args.temperatures, base=2.0)
    run = wandb.init(project=f"thesis-ablation-zoc_temps",
                     entity="wandbefab",
                     name=args.dname)
    zoc_temp_ablation(dset,
                      clip_model,
                      clip_transform,
                      device,
                      args.runs,
                      temperatures)
    run.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--dname", type=str)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--temperatures", type=int, default=10)
    args = parser.parse_args()

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_single(args)


if __name__ == '__main__':
    main()

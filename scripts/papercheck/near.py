import logging

import torch

_logger = logging.getLogger(__name__)


def clip_near_ood_temperatures(clip_model,
                               isolated_classes,
                               id_split,
                               runs,
                               min_temp,
                               max_temp,
                               num_temps,
                               use_origin_templates
                               ):
    import wandb
    from ood_detection.ood_utils import sorted_zeroshot_weights
    from zoc.baseline import get_feature_weight_dict, get_zeroshot_weight_dict

    from datasets.classnames import base_template
    from zeroshot.classification import get_cosine_similarity_matrix_for_normed_features
    import numpy as np
    from zoc.utils import get_auroc_for_max_probs, get_ablation_splits, get_split_specific_targets
    len_all_classes = len(isolated_classes.classes)
    id_classes = int(len_all_classes * id_split)
    ood_classes = len_all_classes - id_classes
    feature_weight_dict = get_feature_weight_dict(isolated_classes, clip_model)

    if use_origin_templates:
        classes_weight_dict = get_zeroshot_weight_dict(isolated_classes, clip_model)
    else:

        isolated_classes.templates = base_template
        classes_weight_dict = get_zeroshot_weight_dict(isolated_classes, clip_model)
    ablation_splits = get_ablation_splits(isolated_classes.classes, n=runs, id_classes=id_classes,
                                          ood_classes=ood_classes)

    # for each temperature..
    for temperature in np.logspace(np.log2(min_temp), np.log2(max_temp), num=num_temps,
                                   base=2.0):  # 10 values between .007 and 100

        split_mlp_aurocs, split_msp_aurocs = [], []
        for split in ablation_splits:

            seen_labels = split[:id_classes]
            unseen_labels = split[id_classes:]
            _logger.debug(f"Seen labels: {seen_labels}\nOOD Labels: {split[id_classes:]}")

            zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)

            mlp, msp = [], []
            f_probs_sum, acc_probs_sum, id_probs_sum = [], [], []

            # do 10 times
            for i, semantic_label in enumerate(split):
                # get features
                image_features_for_label = feature_weight_dict[semantic_label]
                # calc the logits and softmaxs
                zeroshot_probs = get_cosine_similarity_matrix_for_normed_features(image_features_for_label,
                                                                                  zeroshot_weights, temperature)
                softmax_probs = torch.softmax(zeroshot_probs, dim=-1)

                assert zeroshot_probs.shape[1] == id_classes
                # detection score is accumulative sum of probs of generated entities
                # careful, only for this setting axis=1

                top_mlp_prob, _ = zeroshot_probs.cpu().topk(1, dim=-1)
                mlp.extend(top_mlp_prob.detach().numpy())
                top_msp_prob, _ = softmax_probs.cpu().topk(1, dim=-1)
                msp.extend(top_msp_prob.detach().numpy())

            targets = get_split_specific_targets(isolated_classes, seen_labels, unseen_labels)
            mlp_auroc = get_auroc_for_max_probs(targets, mlp)
            msp_auroc = get_auroc_for_max_probs(targets, msp)
            split_mlp_aurocs.append(mlp_auroc)
            split_msp_aurocs.append(msp_auroc)

        result = {'mlp': np.mean(split_mlp_aurocs),
                  'mlp_std': np.std(split_mlp_aurocs),
                  'msp': np.mean(split_msp_aurocs),
                  'msp_std': np.std(split_msp_aurocs),
                  'temperature': temperature}
        wandb.log(result)
    return True


def run_all(args):
    from datasets.zoc_loader import IsolatedClasses

    import clip

    import wandb
    from datasets.config import DATASETS_DICT
    from ood_detection.config import Config
    from tqdm import tqdm

    datasets = DATASETS_DICT
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    device = Config.DEVICE

    for dname, dset in tqdm(datasets.items()):
        isolated_classes = IsolatedClasses(dset(Config.DATAPATH,
                                                transform=clip_transform,
                                                split='test'),
                                           batch_size=512)

        use_templates = bool(args.templates)
        if use_templates:
            project_name = "thesis-near_ood-temperature-ct-temps"
        else:
            project_name = "thesis-near_ood-temperature-dt-temps"

        run = wandb.init(project=project_name,
                         entity="wandbefab",
                         name=dname,
                         config={"runs": args.runs,
                                 "temps": args.templates,
                                 "id_split": Config.ID_SPLIT})
        _logger.info(f"Using origin templates: {use_templates}")
        results = clip_near_ood_temperatures(clip_model,
                                             isolated_classes,
                                             Config.ID_SPLIT,
                                             args.runs,
                                             0.01,
                                             100.,
                                             args.temperatures,
                                             args.templates)
        run.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--temperatures", type=int, default=10)
    parser.add_argument("--templates", type=int, required=True)
    args = parser.parse_args()

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_all(args)


if __name__ == '__main__':
    main()

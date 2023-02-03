import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from src.zoc.utils import get_zoc_unique_entities, tokenize_for_clip, get_zoc_unique_entities
import logging
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import wandb

from adapters.tip_adapter import create_tip_train_set, get_cache_model, get_dataset_with_shorted_classes, \
    get_dataset_features_from_dataset_with_split, run_tip_adapter_finetuned, WeightAdapter, load_adapter, search_hp, \
    get_cache_logits
from datasets.zoc_loader import IsolatedClasses
from ood_detection.ood_utils import sorted_zeroshot_weights
from zoc.baseline import get_feature_weight_dict, get_zeroshot_weight_dict

from adapters.ood import get_ablation_split_classes, pad_list_of_vectors

from clip.simple_tokenizer import SimpleTokenizer
from transformers import BertGenerationTokenizer
from zoc.utils import get_decoder, get_ablation_splits, get_zoc_logits_dict, \
    get_auroc_for_ood_probs, get_auroc_for_max_probs, get_mean_std, get_split_specific_targets
import clip
from ood_detection.config import Config
from datasets.config import DATASETS_DICT

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

run_clearml = False
runs = 10
kshots = [2, 4, 6, 8, 16, 32, 64, 128]
train_epochs = 20
augment_epochs = 10
lr = 0.001
eps = 1e-4


def main():
    failed = []
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    device = Config.DEVICE
    bert_tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    clip_tokenizer = SimpleTokenizer()
    bert_model = get_decoder()

    for dname, dset in DATASETS_DICT.items():
        if dname != 'cifar100':
            continue

        _logger.info(f"\t\tStarting {dname} run...")
        run = wandb.init(project=f"thesis-{dname}_kshot-ablation-{runs}-runs",
                         entity="wandbefab",
                         name=dname)
        try:
            results = adapter_zoc_ablation(dset,
                                           clip_model,
                                           clip_transform,
                                           clip_tokenizer,
                                           bert_tokenizer,
                                           bert_model,
                                           device,
                                           Config.ID_SPLIT,
                                           augment_epochs,
                                           runs,
                                           kshots,
                                           train_epochs,
                                           lr,
                                           eps)
            print(results)
        except Exception as e:
            failed.append(dname)
            raise e
        run.finish()
    print(f"Failed: {failed}")


def adapter_zoc_ablation(dset,
                         clip_model,
                         clip_transform,
                         clip_tokenizer,
                         bert_tokenizer,
                         bert_model,
                         device,
                         id_classes_split,
                         augment_epochs,
                         runs_per_setting,
                         kshots,
                         train_epochs,
                         learning_rate,
                         eps,
                         shorten_classes=None):
    dataset = dset(data_path=Config.DATAPATH,
                   split='test',
                   transform=clip_transform)
    # prepare features ...
    isolated_classes_fast_loader = IsolatedClasses(dataset,
                                                   batch_size=512,
                                                   lsun=False)

    # CAREFUL: ADJUSTMENT FOR ZOC: THE TEMPLATES ( train tip on same )
    isolated_classes_fast_loader.templates = ["This is a photo of a {}"]
    _logger.info('Creating the test weight dicts')
    feature_weight_dict = get_feature_weight_dict(isolated_classes_fast_loader, clip_model, device)
    classes_weight_dict = get_zeroshot_weight_dict(isolated_classes_fast_loader, clip_model)
    _logger.info("Done creating weight dicts.")

    # prepare ablation splits...
    num_id_classes = int(len(dataset.classes) * id_classes_split)
    num_ood_classes = len(dataset.classes) - num_id_classes
    if shorten_classes:
        _logger.warning(f"SHORTENING CLASSES TO {shorten_classes}")
        num_id_classes = int(shorten_classes * Config.ID_SPLIT)
        num_ood_classes = shorten_classes - num_id_classes
    _logger.info(f"ID classes: {num_id_classes}, OOD classes: {num_ood_classes}")

    all_seen_descriptions = [f"This is a photo of a {label}" for label in dataset.classes]
    zoc_unique_entities = get_zoc_unique_entities(dataset, all_seen_descriptions, clip_model, clip_tokenizer,
                                              bert_tokenizer,
                                              bert_model, device)

    isolated_classes_slow_loader = IsolatedClasses(dataset,
                                                   batch_size=1,
                                                   lsun=False)

    for kshot in kshots:

        ablation_splits = get_ablation_splits(dataset.classes, runs_per_setting, num_id_classes, num_ood_classes)

        # run for the ablation splits
        clip_aucs, tip_aucs, tipf_aucs = [], [], []
        zoc_aucs, toc_aucs, tocf_aucs = [], [], []

        for split_idx, split in enumerate(tqdm(ablation_splits)):
            _logger.info(f"Split ({split_idx + 1} / {len(ablation_splits)} )")

            seen_descriptions, seen_labels, unseen_labels = get_ablation_split_classes(num_id_classes, split)

            # prep everything for tip(f)
            zeroshot_weights = sorted_zeroshot_weights(classes_weight_dict, seen_labels)
            zeroshot_weights = zeroshot_weights.to(torch.float32)

            # get the kshot train set
            tip_train_set = create_tip_train_set(dset, seen_labels, kshot)
            tip_train_set.name = f"{tip_train_set.name}_ablation_kshot"
            _logger.info(f"len train set: {len(tip_train_set)}. Should be: {len(tip_train_set.classes) * kshots} (max)")
            cache_keys, cache_values = get_cache_model(tip_train_set, clip_model, augment_epochs=augment_epochs)

            # get shorted val set for the
            tip_val_set = get_dataset_with_shorted_classes(dset, seen_labels, 'val')
            # get features from the shorted val set
            val_features, val_labels, label_features, classes = get_dataset_features_from_dataset_with_split(
                tip_val_set,
                clip_model)

            # set init residual ratio to 1 ( new & old knowledge balanced)
            init_alpha = 1.
            # set sharpness nearly balanced
            init_beta = 1.17
            tipf_alpha, tipf_beta = run_tip_adapter_finetuned(tip_train_set, clip_model,
                                                              val_features, val_labels,
                                                              zeroshot_weights, cache_keys,
                                                              cache_values, init_alpha, init_beta,
                                                              train_epochs, learning_rate,
                                                              eps)
            tipf_adapter = WeightAdapter(cache_keys).to(device)
            tipf_adapter.load_state_dict(load_adapter(tip_train_set.name))
            tipf_adapter.eval()

            tip_alpha, tip_beta = search_hp(cache_keys, cache_values, val_features, val_labels, zeroshot_weights)
            # run zoc
            clip_probs_max, tip_probs_max, tipf_probs_max = [], [], []
            zoc_probs_sum, toc_probs_sum, tocf_probs_sum = [], [], [],

            for label_idx, semantic_label in enumerate(split):

                # get features
                test_image_features_for_label = feature_weight_dict[semantic_label]
                test_image_features_for_label = test_image_features_for_label.to(torch.float32)

                # calc the logits and softmax
                clip_logits = 100 * test_image_features_for_label @ zeroshot_weights.T
                clip_probs = torch.softmax(clip_logits, dim=-1).squeeze()
                top_clip_prob, _ = clip_probs.cpu().topk(1, dim=-1)
                clip_probs_max.extend(top_clip_prob.detach().numpy())

                if clip_probs.shape[1] != num_id_classes:
                    _logger.error(f"Z_p.shape: {clip_probs.shape} != id: {num_id_classes}")
                    raise AssertionError

                # ZOC
                zoc_entities_for_semantic_label = zoc_unique_entities[semantic_label]

                zoc_logits_for_semantic_label = []

                loader = isolated_classes_slow_loader[semantic_label]
                for image, unique_entities in zip(loader, zoc_entities_for_semantic_label):
                    all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in unique_entities]
                    all_desc_ids = tokenize_for_clip(all_desc, clip_tokenizer)
                    with torch.no_grad():
                        image_feature = clip_model.encode_image(image.to(device)).float()
                        image_feature /= image_feature.norm(dim=-1, keepdim=True)
                        image_feature = image_feature.to(torch.float32)
                        text_features = clip_model.encode_text(all_desc_ids.to(device)).to(torch.float32)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                    zoc_logits_for_image = (100.0 * image_feature @ text_features.T).squeeze().cpu()
                    zoc_entities_for_semantic_label.append(zoc_logits_for_image)
                    zoc_probs = torch.softmax(zoc_logits_for_image, dim=0)
                    zoc_probs_sum.append(torch.sum(zoc_probs[len(seen_labels):]))  # for normal zoc

                # now: use normal zoc probs. use zoctip. use zoctipf

                # first, pad all to then longest with -inf (neutral element in softmax)
                padded_zoc_logits_for_semantic_label = pad_list_of_vectors(zoc_logits_for_semantic_label, -np.inf)

                # TIPF ADAPTER
                tipf_affinity = tipf_adapter(test_image_features_for_label)
                tipf_cache_logits = get_cache_logits(tipf_affinity, cache_values, tipf_beta)
                tipf_logits = clip_logits + tipf_cache_logits * tipf_alpha
                tipf_probs = torch.softmax(tipf_logits, dim=1).squeeze()
                top_tipf_prob, _ = tipf_probs.cpu().topk(1, dim=-1)
                tipf_probs_max.extend(top_tipf_prob.detach().numpy())

                # tip
                tip_affinity = test_image_features_for_label @ cache_keys
                tip_cache_logits = get_cache_logits(tip_affinity, cache_values, tip_beta)
                tip_logits = clip_logits + tip_cache_logits * tip_alpha
                tip_probs = torch.softmax(tip_logits, dim=1).squeeze()
                top_tip_prob, _ = tip_probs.cpu().topk(1, dim=-1)
                tip_probs_max.extend(top_tip_prob.detach().numpy())

                # zoc tip
                padded_cache_logits = torch.zeros(padded_zoc_logits_for_semantic_label.shape)
                padded_cache_logits[:, :tip_cache_logits.shape[1]] = tip_cache_logits
                # the magic
                toc_logits = padded_zoc_logits_for_semantic_label + padded_cache_logits * tip_alpha
                toc_probs = torch.softmax(toc_logits, dim=1).squeeze()
                toc_probs_sum.extend(torch.sum(toc_probs[:, len(seen_labels):], dim=1).detach().numpy())

                # zoc tipf
                padded_cache_logits = torch.zeros(padded_zoc_logits_for_semantic_label.shape)
                padded_cache_logits[:, :tipf_cache_logits.shape[1]] = tipf_cache_logits
                # the magic
                tocf_logits = padded_zoc_logits_for_semantic_label + padded_cache_logits * tipf_alpha
                tocf_probs = torch.softmax(tocf_logits, dim=1).squeeze()
                tocf_probs_sum.extend(torch.sum(tocf_probs[:, len(seen_labels):], dim=1).detach().numpy())

            targets = get_split_specific_targets(isolated_classes_fast_loader, seen_labels, unseen_labels)

            assert len(targets) == len(zoc_probs_sum), f"{len(targets)} != {len(zoc_probs_sum)}"
            assert len(targets) == len(clip_probs_max), f"{len(targets)} != {len(clip_probs_max)}"
            assert len(targets) == len(toc_probs_sum), f"{len(targets)} != {len(tocf_probs_sum)}"
            assert len(targets) == len(tocf_probs_sum), f"{len(targets)} != {len(tocf_probs_sum)}"
            assert len(targets) == len(tip_probs_max), f"{len(targets)} != {len(tip_probs_max)}"
            assert len(targets) == len(tipf_probs_max), f"{len(targets)} != {len(tipf_probs_max)}"

            clip_aucs.append(get_auroc_for_max_probs(targets, np.array(clip_probs_max)))
            tip_aucs.append(get_auroc_for_max_probs(targets, tip_probs_max))
            tipf_aucs.append(get_auroc_for_max_probs(targets, tipf_probs_max))
            zoc_aucs.append(get_auroc_for_ood_probs(targets, zoc_probs_sum))
            toc_aucs.append(get_auroc_for_ood_probs(targets, toc_probs_sum))
            tocf_aucs.append(get_auroc_for_ood_probs(targets, tocf_probs_sum))

        # summed up over splits
        clip_mean, clip_std = get_mean_std(clip_aucs)
        tip_mean, tip_std = get_mean_std(tip_aucs)
        tipf_mean, tipf_std = get_mean_std(tipf_aucs)

        zoc_mean, zoc_std = get_mean_std(zoc_aucs)
        toc_mean, toc_std = get_mean_std(toc_aucs)
        tocf_mean, tocf_std = get_mean_std(tocf_aucs)

        metrics = {'clip': clip_mean,
                   'clip_std': clip_std,
                   'tip': tip_mean,
                   'tip_std': tip_std,
                   'tipf': tipf_mean,
                   'tipf_std': tipf_std,
                   'zoc': zoc_mean,
                   'zoc_std': zoc_std,
                   'toc': toc_mean,
                   'toc_std': toc_std,
                   'tocf': tocf_mean,
                   'tocf_std': tocf_std,
                   'shots': kshot
                   }
        wandb.log(metrics)


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

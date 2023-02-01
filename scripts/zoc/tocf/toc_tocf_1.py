import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from clip.simple_tokenizer import SimpleTokenizer
from transformers import BertGenerationTokenizer
from zoc.utils import get_decoder

from adapters.ood import adapter_zoc
import clip
from ood_detection.config import Config
import logging

import wandb
from datasets.config import DATASETS_DICT

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

run_clearml = False
runs = 10
kshots = 16
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
        if dname not in ['caltech101', 'caltech cub']:
            continue
        # if dname not in ['cifar10', 'cifar100']:
        #     continue
        # if dname not in ['dtd', 'fashion mnist']:
        #     continue
        # if dname not in ['flowers102', 'gtsrb']:
        #     continue
        # if dname not in ['imagenet', 'mnist']:
        #     continue
        # if dname not in ['stanford cars', 'svhn']:
        #     continue
        _logger.info(f"\t\tStarting {dname} run...")
        run = wandb.init(project=f"thesis-toc-ood-test-hyperparam-search-{runs}-runs",
                         entity="wandbefab",
                         name=dname)
        try:
            results = adapter_zoc(dset,
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
        wandb.log(results)
        run.finish()
    print(f"Failed: {failed}")


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

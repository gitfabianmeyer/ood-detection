import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import logging
import wandb
from zoc.ablation import kshot_adapter_zoc_ablation

from clip.simple_tokenizer import SimpleTokenizer
from transformers import BertGenerationTokenizer
from zoc.utils import get_decoder
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
            results = kshot_adapter_zoc_ablation(dset,
                                                 clip_model,
                                                 clip_transform,
                                                 clip_tokenizer,
                                                 bert_tokenizer,
                                                 bert_model,
                                                 Config.ID_SPLIT,
                                                 augment_epochs,
                                                 runs,
                                                 kshots,
                                                 train_epochs,
                                                 lr,
                                                 eps)
            wandb.log(results)
        except Exception as e:
            failed.append(dname)
            raise e
        run.finish()
    print(f"Failed: {failed}")


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

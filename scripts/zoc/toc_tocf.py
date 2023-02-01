import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from clip.simple_tokenizer import SimpleTokenizer
from transformers import BertGenerationTokenizer
from zoc.utils import get_decoder

from adapters.ood import adapter_zoc # TODO
import clip
from ood_detection.config import Config
import logging

import wandb
from datasets.config import HalfTwoDict, HalfOneDict

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

run_clearml = False
runs = 1 # TODO
kshots = 2
train_epochs = 1
augment_epochs = 1
lr = 0.001
eps = 1e-4


def main():
    failed = []
    clip_model, clip_transform = clip.load(Config.VISION_MODEL)
    device = Config.DEVICE
    bert_tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
    clip_tokenizer = SimpleTokenizer()
    bert_model = get_decoder()

    for dname, dset in HalfOneDict.items():
        if dname != 'dtd':
            continue
        _logger.info(f"\t\tStarting {dname} run...")
        # run = wandb.init(project=f"thesis-toc-ood-test-hyperparam-search-{runs}-runs", # TODO
        #                  entity="wandbefab",
        #                  name=dname)
        results = adapter_zoc(dset,
                              clip_model,
                              clip_transform,
                              clip_tokenizer,
                              bert_tokenizer,
                              bert_model,
                              device,
                              .05,
                              #Config.ID_SPLIT, # TODO
                              augment_epochs,
                              runs,
                              kshots,
                              train_epochs,
                              lr,
                              eps)
        print(results)

        #
        # wandb.log(results)
        # run.finish()
    print(f"Failed: {failed}")


if __name__ == '__main__':

    if run_clearml:
        from clearml import Task

        print("running clearml")
        task = Task.init(project_name="ma_fmeyer", task_name="tip adapter testing")
        task.execute_remotely('5e62040adb57476ea12e8593fa612186')
    main()

import clip
from datasets.cifar import OodCifar10

from src.adapters.tip_adapter import get_cache_model, create_tip_train_set
from src.datasets.fashion_mnist import OodFashionMNIST
from src.datasets.mnist import OodMNIST
from src.datasets.svhn import OodSVHN
from src.ood_detection.config import Config

dset = OodSVHN
# dset = OodCifar10
clip_model, clip_transform = clip.load(Config.VISION_MODEL)
seen_labels = ['0 - zero', '1 - one', '2 - two', '3 - three',
               '4 - four', ]
# seen_labels = ['airplane', 'automobile', 'bird']
tip_train_set = create_tip_train_set(dset, seen_labels, 4)
cache_keys, cache_values = get_cache_model(tip_train_set, clip_model, augment_epochs=1)

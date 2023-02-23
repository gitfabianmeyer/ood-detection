import torch

from src.datasets.config import DATASETS_DICT
from src.ood_detection.config import Config
from src.zeroshot.utils import FeatureDict
from src.zoc.utils import get_zoc_feature_dict

# for dname, dset in DATASETS_DICT.items():
#     print(f"instantiating {dname}")
#     dataset = dset(Config.DATAPATH,
#                    transform=None,
#                    split='test')
#     clip_model = None
#     fd = FeatureDict(dataset, clip_model)


labels = [i for i in range(2)]
fake_features = {i: torch.rand((i + 3, 4)) for i in labels}

fd = FeatureDict(fake_features, None)
print("success")
d = get_zoc_feature_dict(fd, None)
for key, value in d.items():
    print(key)
    print(value.shape)

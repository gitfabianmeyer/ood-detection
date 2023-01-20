import os
import torch


class Config:
    DATAPATH = '/mnt/c/Users/fmeyer/Git/ood-detection/data'
    PLOTS = os.path.join(DATAPATH, 'plots')
    FEATURES = os.path.join(DATAPATH, 'features')
    VISION_MODEL = 'ViT-B/32'
    MODELS = os.path.join(DATAPATH, "models")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TEST_SIZE = .4
    ID_SPLIT = .4

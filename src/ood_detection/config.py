import os
import torch


class Config:
    DATAPATH = '/mnt/c/Users/fmeyer/Git/ood-detection/data'
    PLOTS = os.path.join(DATAPATH, 'plots')
    FEATURES = os.path.join(DATAPATH, 'features')
    VISION_MODEL = 'ViT-B/32'
    MODELS = os.path.join(DATAPATH, "models")
    gpu = 1
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    DEVICE = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


import os.path


class Config:
    DATAPATH = '/mnt/c/Users/fmeyer/Git/ood-detection/data'
    PLOTS = os.path.join(DATAPATH, 'plots')
    FEATURES = os.path.join(DATAPATH, 'features')
    VISION_MODEL = 'ViT-B/32'
    MODELS = os.path.join(DATAPATH, "models")

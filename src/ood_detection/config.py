import os.path


class Config:
    DATAPATH = '/mnt/c/Users/fmeyer/Git/ood-detection/data'
    PLOTS = os.path.join(DATAPATH, 'plots')
    FEATURES = os.path.join(DATAPATH, 'features')
    VISION_MODEL = 'ViT-L/14@336px'
    MODELS = os.path.join(DATAPATH, "models")

import torch
import os
import typing
import tqdm


class DatasetPINGP(torch.utils.data.Dataset):
    def __init__(self, dataset_path, downscale: int):
        import yaml
        with open(os.path.join(dataset_path, 'scene_info.yaml'), 'r') as f:
            scene_info = yaml.safe_load(f)

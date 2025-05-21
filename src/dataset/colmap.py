import torch
import typing
import tqdm
import os


class DatasetColmap(torch.utils.data.Dataset):
    def __init__(self, dataset_path, downscale: int):
        import json
        with open(os.path.join(dataset_path, 'transforms.json'), 'r') as f:
            transform = json.load(f)

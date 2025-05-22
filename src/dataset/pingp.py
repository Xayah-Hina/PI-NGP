import torch
import os
import typing
import tqdm


class DatasetPINGP(torch.utils.data.Dataset):
    def __init__(self, dataset_path, downscale: int):
        import yaml
        with open(os.path.join(dataset_path, 'scene_info.yaml'), 'r') as f:
            scene_info = yaml.safe_load(f)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return {}

    def __str__(self):
        fields = f"Fields[" + ", ".join(f"{k}" for k, v in self.__dict__.items()) + "]"
        return fields

    def plot(self, index: int):
        from matplotlib import pyplot as plt
        image = self.images[index]
        plt.imshow(image)
        plt.show()

from ..spatial import get_rays
from .blender import DatasetBlender
from .colmap import DatasetColmap
from .pingp import DatasetPINGP
import dataclasses
import torch
import os
import typing


@dataclasses.dataclass
class NeRFDatasetConfig:
    # required options
    dataset_dir: str = dataclasses.field(metadata={'help': 'base directory of dataset'})

    # optional options
    data_dir: str = dataclasses.field(default=os.path.abspath(os.path.join(os.getcwd(), 'data')), metadata={'help': 'data directory'})

    camera_radius_scale: float = dataclasses.field(default=0.33, metadata={'help': 'scale camera location into box[-bound, bound]^3'})
    camera_offset: list = dataclasses.field(default_factory=lambda: [0, 0, 0], metadata={'help': 'offset of camera location'})

    use_error_map: bool = dataclasses.field(default=False, metadata={'help': 'use error map to sample rays'})
    use_preload: bool = dataclasses.field(default=True, metadata={'help': 'preload all data into GPU, accelerate training but use more GPU memory'})
    use_fp16: bool = dataclasses.field(default=True, metadata={'help': 'use amp mixed precision training'})

    downscale: int = dataclasses.field(default=1, metadata={'help': 'downscale factor for images'})
    color_space: str = dataclasses.field(default='srgb', metadata={'help': 'Color space, supports (linear, srgb)'})
    num_rays: int = dataclasses.field(default=4096, metadata={'help': 'number of rays to sample per image'})

    device: str = dataclasses.field(default='cuda:0', metadata={'help': 'device to use, usually setting to None is OK. (auto choose device)'})

    def __post_init__(self):
        if not os.path.exists(os.path.join(self.data_dir, self.dataset_dir)):
            raise FileNotFoundError(f"Dataset directory {self.data_dir}/{self.dataset_dir} does not exist.")


class NeRFDataset:
    def __init__(self, config: NeRFDatasetConfig, dataset_type: typing.Literal['train', 'val', 'test']):
        base_dataset_dir = os.path.join(config.data_dir, config.dataset_dir)
        if os.path.exists(os.path.join(base_dataset_dir, 'transforms.json')):
            raise NotImplementedError('[NOT IMPLEMENTED] DatasetColmap')
        elif os.path.exists(os.path.join(base_dataset_dir, 'transforms_train.json')):
            self.mode = 'blender'
            self.dataset = DatasetBlender(
                dataset_path=base_dataset_dir,
                dataset_type=dataset_type,
                downscale=config.downscale,
                camera_radius_scale=config.camera_radius_scale,
                camera_offset=config.camera_offset,
                use_error_map=config.use_error_map,
                use_preload=config.use_preload,
                use_fp16=config.use_fp16,
                color_space=config.color_space,
                device=torch.device(config.device),
            )
        elif os.path.exists(os.path.join(base_dataset_dir, 'scene_info.yaml')):
            self.mode = 'pi-ngp'
            raise NotImplementedError('[NOT IMPLEMENTED] DatasetPINGP')
        else:
            raise NotImplementedError('[INVALID DATASET TYPE] NeRFDataset at: {}'.format(base_dataset_dir))
        self.dataset_type = dataset_type
        self.num_rays = config.num_rays if dataset_type == 'train' else -1
        self.device = torch.device(config.device)

    def collate(self, batch: list):
        B = len(batch)  # a list of length 1
        poses, width, height, intrinsics = torch.stack([item['pose'] for item in batch], dim=0), self.dataset.width, self.dataset.height, self.dataset.intrinsics
        error_maps = torch.stack([item['error_map'] for item in batch], dim=0) if self.dataset.error_map is not None else None

        poses = poses.to(self.device)  # [B, 4, 4]
        rays = get_rays(poses, intrinsics, height, width, self.num_rays, error_maps)

        times = torch.stack([item['time'] for item in batch], dim=0) if self.dataset.times is not None else None
        if times is not None:
            times = times.to(self.device)  # [B, 1]

        images = torch.stack([item['image'] for item in batch], dim=0) if self.dataset.images is not None else None
        if images is not None:
            images = images.to(self.device)  # [B, H, W, 3/4]
            if self.dataset_type == 'train':
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
        results = {
            'images': images,
            'time': times,
            'H': height,
            'W': width,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'color_space': self.dataset.color_space,
        }
        return results

    def dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=1,
            collate_fn=self.collate,
            shuffle=True,
            num_workers=0,
        )

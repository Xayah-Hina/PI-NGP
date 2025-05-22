import torch
import torchvision.io as io
import os
import typing
import tqdm


class DatasetPINGP(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path,
                 dataset_type: typing.Literal['train', 'val', 'test'],
                 downscale: int,
                 use_preload: bool,
                 use_fp16: float,
                 device: torch.device,
                 ):
        import numpy as np
        import yaml

        self.dtype = torch.half if use_fp16 else torch.float

        with open(os.path.join(dataset_path, 'scene_info.yaml'), 'r') as f:
            scene_info = yaml.safe_load(f)
            if dataset_type == 'train' or dataset_type == 'val':
                videos_info = scene_info['training_videos'] if dataset_type == 'train' else scene_info['validation_videos']
                cameras_info = scene_info['training_camera_calibrations'] if dataset_type == 'train' else scene_info['validation_camera_calibrations']

                _frames_tensors = []
                for _path in tqdm.tqdm([os.path.normpath(os.path.join(dataset_path, video_path)) for video_path in videos_info], desc=f'[Loading {self.__class__.__name__}...] ({dataset_type})'):
                    try:
                        _frames, _, _ = io.read_video(_path, pts_unit='sec')
                        _frames = _frames.to(dtype=self.dtype) / 255.0
                        _frames_tensors.append(_frames)
                    except Exception as e:
                        print(f"Error loading video '{_path}': {e}")
                videos = torch.stack(_frames_tensors)

                V, T, H, W, C = videos.shape
                videos_permuted = videos.permute(0, 1, 4, 2, 3).reshape(V * T, C, H, W)
                new_H, new_W = int(H // downscale), int(W // downscale)
                videos_resampled = torch.nn.functional.interpolate(videos_permuted, size=(new_H, new_W), mode='bilinear', align_corners=False)
                self.images = videos_resampled.reshape(V, T, C, new_H, new_W).permute(1, 0, 3, 4, 2)

                camera_infos = [np.load(path) for path in [os.path.normpath(os.path.join(dataset_path, camera_path)) for camera_path in cameras_info]]
                self.width = set([int(info["width"]) for info in camera_infos]).pop()
                self.height = set([int(info["height"]) for info in camera_infos]).pop()
                self.poses = torch.stack([torch.tensor(info["cam_transform"], device=device, dtype=torch.float) for info in camera_infos])
                focal = set([info["focal"] * self.width / info["aperture"] for info in camera_infos]).pop()
                self.intrinsics = torch.tensor([focal, focal, self.width / 2, self.height / 2])

            elif dataset_type == 'test':
                raise NotImplementedError('[PINGP] Test dataset not supported yet.')
            else:
                raise ValueError(f"Invalid dataset type: {dataset_type}. Expected 'train', 'val', or 'test'.")

        if use_preload:
            if self.images is not None:
                self.images = self.images.to(self.dtype).to(device)
            self.poses = self.poses.to(device)
            # if self.times is not None:
            #     self.times = self.times.to(device)

        info = f"{self.__class__.__name__}({len(self.images) if self.images is not None else 0} images, {self.width}x{self.height}, fl_x={self.intrinsics[0]:.2f}, fl_y={self.intrinsics[1]:.2f}, cx={self.intrinsics[2]:.2f}, cy={self.intrinsics[3]:.2f}, {self.dtype})"
        print(f"Loaded: {info}")

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        # TODO: interpolate fractural time
        return {
            'image': self.images[idx] if self.images is not None else None,
            'pose': self.poses[idx],
            'time': self.times[idx] if self.times is not None else None,
            'error_map': self.error_map[idx] if self.error_map is not None else None,
        }

    def __str__(self):
        fields = f"Fields[" + ", ".join(f"{k}" for k, v in self.__dict__.items()) + "]"
        return fields

    def plot(self, index: int):
        from matplotlib import pyplot as plt
        image = self.images[index]
        plt.imshow(image)
        plt.show()

import torch
import typing
import tqdm
import os


class DatasetBlender(torch.utils.data.Dataset):
    def __init__(self, dataset_path, dataset_type: typing.Literal['train', 'val', 'test'], downscale: int, camera_radius_scale: float, camera_offset: list, use_error_map: bool, use_preload: bool, use_fp16: float, color_space: str, device: torch.device):
        import json
        import cv2
        import numpy as np

        # ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
        def nerf_matrix_to_ngp(pose, scale: float, offset: list):
            # for the fox dataset, 0.33 scales camera radius to ~ 2
            new_pose = np.array([
                [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
                [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
                [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            return new_pose

        images = []
        poses = []
        times = []
        widths = []
        heights = []
        with open(os.path.join(dataset_path, 'transforms_' + dataset_type + '.json'), 'r') as json_file:
            transform = json.load(json_file)
            for f in tqdm.tqdm(transform["frames"], desc=f'[Loading {self.__class__.__name__}...] ({dataset_type})'):
                image = cv2.imread(dataset_path + f['file_path'] + '.png', cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                if downscale > 1:
                    image = cv2.resize(image, (image.shape[0] // downscale, image.shape[1] // downscale), interpolation=cv2.INTER_AREA)
                image = image.astype(np.float32) / 255.0
                widths.append(image.shape[1])
                heights.append(image.shape[0])
                images.append(image)
                poses.append(nerf_matrix_to_ngp(np.array(f['transform_matrix'], dtype=np.float32), scale=camera_radius_scale, offset=camera_offset))
                if 'time' in f:
                    times.append(f['time'])
            assert len(set(widths)) == 1 and len(set(heights)) == 1, '[INVALID DATASET IMAGES SIZE] All images must have the same size'
            self.width = set(widths).pop()  # int
            self.height = set(widths).pop()  # int
            if 'fl_x' in transform or 'fl_y' in transform:
                fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
                fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
            elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
                # blender, assert in radians. already downscaled since we use H/W
                fl_x = self.width / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
                fl_y = self.height / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
                if fl_x is None: fl_x = fl_y
                if fl_y is None: fl_y = fl_x
            else:
                raise RuntimeError('Failed to load focal length, please check the transforms.json!')
            cx = (transform['cx'] / downscale) if 'cx' in transform else (self.width / 2)
            cy = (transform['cy'] / downscale) if 'cy' in transform else (self.height / 2)
            self.intrinsics = torch.tensor([fl_x, fl_y, cx, cy])

        self.images = torch.from_numpy(np.stack(images, axis=0))  # [N, H, W, C]
        self.poses = torch.from_numpy(np.stack(poses, axis=0))  # [N, 4, 4]
        self.times = torch.from_numpy(np.asarray(times, dtype=np.float32)).view(-1, 1) if len(times) > 0 else None
        self.dtype = torch.half if (use_fp16 and color_space != 'linear') else torch.float
        self.color_space = color_space

        # manual normalize
        if self.times is not None and self.times.max() > 1:
            self.times = self.times / (self.times.max() + 1e-8)  # normalize to [0, 1]

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        if use_error_map and dataset_type == 'train':
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float)  # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        if use_preload:
            self.images = self.images.to().to(device)
            self.poses = self.poses.to(device)
            if self.times is not None:
                self.times = self.times.to(device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(device)

        info = f"{self.__class__.__name__}({len(self.images)} images, {self.width}x{self.height}, {self.color_space}, radius={self.radius:.2f}, fl_x={self.intrinsics[0]:.2f}, fl_y={self.intrinsics[1]:.2f}, cx={self.intrinsics[2]:.2f}, cy={self.intrinsics[3]:.2f}, {self.dtype})"
        print(f"Loaded: {info}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
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

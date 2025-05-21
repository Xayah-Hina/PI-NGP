from .network import NeRFNetworkNGP, NeRFNetworkBasis, NeRFRendererStatic, NeRFRendererDynamic
from .dataset import NeRFDataset
import torch
import torch.utils.tensorboard
import torch_ema
import tqdm
import dataclasses
import typing
import os


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@dataclasses.dataclass
class TrainerConfig:
    # required options
    model: typing.Literal["ngp", "basis"] = dataclasses.field(metadata={"help": "model type"})

    # optional options
    name: str = dataclasses.field(default="default name", metadata={"help": "name of the experiment"})
    workspace: str = dataclasses.field(default="workspace", metadata={"help": "workspace directory"})
    mode: typing.Literal["train", "test"] = dataclasses.field(default="train", metadata={"help": "mode of training"})

    lr_encoding: float = dataclasses.field(default=1e-2, metadata={"help": "initial learning rate for encoding"})
    lr_net: float = dataclasses.field(default=1e-3, metadata={"help": "initial learning rate for network"})
    ema_decay: float = dataclasses.field(default=0.95, metadata={"help": "decay rate for exponential moving average"})

    use_fp16: bool = dataclasses.field(default=True, metadata={"help": "use amp mixed precision training"})
    device: str = dataclasses.field(default="cuda:0", metadata={"help": "device to use, usually setting to None is OK. (auto choose device)"})

    dt_gamma: float = dataclasses.field(default=0, metadata={"help": "dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)"})


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.name = config.name
        self.workspace = config.workspace
        if config.model == 'ngp':
            self.model = NeRFNetworkNGP(
                encoding_spatial='hashgrid',
                encoding_dir='sphere_harmonics',
                encoding_bg='hashgrid',
            ).to(config.device)
        elif config.model == 'basis':
            self.model = NeRFNetworkBasis(
                encoding_spatial='tiledgrid',
                encoding_dir='sphere_harmonics',
                encoding_time='frequency',
                encoding_bg='hashgrid',
            ).to(config.device)
        else:
            raise NotImplementedError(f"Model {config.model} not implemented")
        self.optimizer = torch.optim.Adam(self.model.get_params(config.lr_encoding, config.lr_net), betas=(0.9, 0.99), eps=1e-15)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda iter: 0.1 ** min(iter / 30000, 1))
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.use_fp16)
        self.ema = torch_ema.ExponentialMovingAverage(self.model.parameters(), decay=config.ema_decay)
        self.writer = torch.utils.tensorboard.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        self.error_map = None  # Placeholder for error map, if needed
        self.use_fp16 = config.use_fp16
        self.device = config.device

        self.epoch = 0
        self.global_step = 0
        self.local_step = 0

        self.runtime_options = {
            'dt_gamma': config.dt_gamma,
        }

        self.load_checkpoint(checkpoint=None)

    def save_checkpoint(self, full: bool, best: bool):
        save_path = os.path.join(self.workspace, 'checkpoints')
        os.makedirs(save_path, exist_ok=True)
        name = f'{self.name}_ep{self.epoch:04d}'

        state: dict[str, typing.Any] = {
            'epoch': self.epoch,
            'global_step': self.global_step,
        }

        if self.model.runtime_params['cuda_ray']:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:
            state['model'] = self.model.state_dict()
            file_path = f"{save_path}/{name}.pth"
            torch.save(state, file_path)

    def load_checkpoint(self, checkpoint):
        if checkpoint is None:
            import glob
            checkpoint_list = sorted(glob.glob(f'{os.path.join(self.workspace, "checkpoints")}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                print(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                print("[WARN] No checkpoint found, model randomly initialized.")
                return
        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print("[INFO] loaded model.")
        if len(missing_keys) > 0:
            print(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[WARN] unexpected keys: {unexpected_keys}")

    def train(self, train_dataset: NeRFDataset, valid_dataset: NeRFDataset, max_epochs: int):

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.runtime_params['cuda_ray']:
            self.model.mark_untrained_grid(train_dataset.dataset.poses, train_dataset.dataset.intrinsics)

        self.error_map = train_dataset.dataset.error_map

        self.model.train()
        train_loader = train_dataset.dataloader()
        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch + 1

            total_loss = 0
            self.local_step = 0
            for i, data in enumerate(tqdm.tqdm(train_loader)):
                self.global_step += 1
                self.local_step += 1
                # update grid every 16 steps
                if self.model.runtime_params['cuda_ray'] and self.global_step % 200 == 0:
                    with torch.amp.autocast('cuda', enabled=self.use_fp16):
                        self.model.update_extra_state()

                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=self.use_fp16):
                    if isinstance(self.model, NeRFRendererStatic):
                        preds, truths, loss = self.train_step_static(data)
                    elif isinstance(self.model, NeRFRendererDynamic):
                        preds, truths, loss = self.train_step_dynamics(data)
                    else:
                        raise NotImplementedError(f"Model {self.model.__class__.__name__} not implemented")

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.lr_scheduler.step()
                loss_val = loss.item()
                total_loss += loss_val

                self.writer.add_scalar("train/loss", loss_val, self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

            if self.ema is not None:
                self.ema.update()

            average_loss = total_loss / self.local_step
            print(f"Epoch {self.epoch}, Loss: {average_loss:.4f}")

        self.save_checkpoint(full=True, best=False)

    def test(self, test_dataset: NeRFDataset):
        import numpy as np
        import imageio
        import cv2
        save_path = os.path.join(self.workspace, 'results')
        os.makedirs(save_path, exist_ok=True)
        name = f'{self.name}_ep{self.epoch:04d}'

        all_rgb_side_by_side = []
        all_preds_depth = []

        self.model.eval()
        test_loader = test_dataset.dataloader()
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(test_loader)):
                data: dict
                with torch.amp.autocast('cuda', enabled=self.use_fp16):
                    if isinstance(self.model, NeRFRendererStatic):
                        preds, preds_depth, gt_rgb = self.test_step_static(data, bg_color=None)
                    elif isinstance(self.model, NeRFRendererDynamic):
                        preds, preds_depth, gt_rgb = self.test_step_dynamics(data, bg_color=None)
                    else:
                        raise NotImplementedError(f"Model {self.model.__class__.__name__} not implemented")
                if data['color_space'] == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                gt = gt_rgb[0].detach().cpu().numpy()
                gt = (gt * 255).astype(np.uint8)

                pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
                gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
                side_by_side = np.concatenate([gt_bgr, pred_bgr], axis=1)  # 左GT，右Pred

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), side_by_side)
                cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                all_rgb_side_by_side.append(side_by_side)
                all_preds_depth.append(pred_depth)

        imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_rgb_side_by_side, fps=25, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

    def train_step_static(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        images = data['images']  # [B, N, 3/4]
        color_space = data['color_space']
        B, N, C = images.shape

        if color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.runtime_params['bg_radius'] > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            # bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            # bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3])  # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render_static(
            rays_o=rays_o,
            rays_d=rays_d,
            dt_gamma=self.runtime_options['dt_gamma'],
            bg_color=bg_color,
            perturb=True,
            force_all_rays=False,
            max_steps=1024,
            T_thresh=1e-4,
        )

        pred_rgb = outputs['image']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)  # [B, N, 3] --> [B, N]

        # update error_map
        if self.error_map is not None:
            raise NotImplementedError("Error map update not implemented")

        loss = loss.mean()

        return pred_rgb, gt_rgb, loss

    def train_step_dynamics(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        time = data['time']  # [B, 1]
        images = data['images']  # [B, N, 3/4]
        color_space = data['color_space']
        B, N, C = images.shape

        if color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.runtime_params['bg_radius'] > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            # bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            # bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3])  # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render_dynamics(
            rays_o=rays_o,
            rays_d=rays_d,
            time=time,
            dt_gamma=self.runtime_options['dt_gamma'],
            bg_color=bg_color,
            perturb=True,
            force_all_rays=False,
            max_steps=1024,
        )

        pred_rgb = outputs['image']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)  # [B, N, 3] --> [B, N]

        # update error_map
        if self.error_map is not None:
            raise NotImplementedError("Error map update not implemented")

        loss = loss.mean()

        # deform regularization
        if 'deform' in outputs and outputs['deform'] is not None:
            raise NotImplementedError("Deform regularization not implemented")

        return pred_rgb, gt_rgb, loss

    def test_step_static(self, data, bg_color):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        else:
            bg_color = 1

        outputs = self.model.render_static(
            rays_o=rays_o,
            rays_d=rays_d,
            dt_gamma=self.runtime_options['dt_gamma'],
            bg_color=bg_color,
            perturb=True,
            force_all_rays=False,
            max_steps=1024,
            T_thresh=1e-4,
        )

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        gt_rgb = None
        if 'images' in data:
            images = data['images']
            color_space = data['color_space']
            B, H, W, C = images.shape
            if color_space == 'linear':
                images[..., :3] = srgb_to_linear(images[..., :3])
            if C == 4:
                gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
            else:
                gt_rgb = images

        return pred_rgb, pred_depth, gt_rgb

    def test_step_dynamics(self, data, bg_color):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        time = data['time']  # [B, 1]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render_dynamics(
            rays_o=rays_o,
            rays_d=rays_d,
            time=time,
            dt_gamma=self.runtime_options['dt_gamma'],
            bg_color=bg_color,
            perturb=True,
            force_all_rays=False,
            max_steps=1024,
        )

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        gt_rgb = None
        if 'images' in data:
            images = data['images']
            color_space = data['color_space']
            B, H, W, C = images.shape
            if color_space == 'linear':
                images[..., :3] = srgb_to_linear(images[..., :3])
            if C == 4 and bg_color is not None:
                gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
            else:
                gt_rgb = images

        return pred_rgb, pred_depth, gt_rgb

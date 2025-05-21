from .network import NeRFNetworkBasis
from .provider import NeRFDataset
import torch
import torch_ema
import dataclasses
import typing
import tqdm


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@dataclasses.dataclass
class TrainerConfig:
    mode: typing.Literal["train", "test"] = dataclasses.field(default="train", metadata={"help": "mode of training"})
    name: str = dataclasses.field(default="default name", metadata={"help": "name of the experiment"})
    model: typing.Literal["dnerf", "nerf", "mipnerf"] = dataclasses.field(default="dnerf", metadata={"help": "model type"})

    lr_encoding: float = dataclasses.field(default=1e-2, metadata={"help": "initial learning rate for encoding"})
    lr_net: float = dataclasses.field(default=1e-3, metadata={"help": "initial learning rate for network"})
    ema_decay: float = dataclasses.field(default=0.95, metadata={"help": "decay rate for exponential moving average"})

    use_fp16: bool = dataclasses.field(default=False, metadata={"help": "use amp mixed precision training"})
    device: str = dataclasses.field(default="cuda:0", metadata={"help": "device to use, usually setting to None is OK. (auto choose device)"})

    ### runtime options
    dt_gamma: float = dataclasses.field(default=1 / 128, metadata={"help": "dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)"})


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.name = config.name
        if config.model == 'dnerf':
            self.model = NeRFNetworkBasis(
                encoding_spatial='tiledgrid',
                encoding_dir='sphere_harmonics',
                encoding_time='frequency',
                encoding_bg='hashgrid',
                bound=1,
            )
        else:
            raise NotImplementedError(f"Model {config.model} not implemented")
        self.model.to(config.device)
        self.optimizer = torch.optim.Adam(self.model.get_params(config.lr_encoding, config.lr_net), betas=(0.9, 0.99), eps=1e-15)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # TODO: implement a proper scheduler
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.use_fp16)
        self.ema = torch_ema.ExponentialMovingAverage(self.model.parameters(), decay=config.ema_decay)
        self.error_map = None  # Placeholder for error map, if needed
        self.use_fp16 = config.use_fp16
        self.device = config.device

        self.epoch = 0
        self.global_step = 0
        self.local_step = 0

        self.runtime_options = {
            'dt_gamma': config.dt_gamma,
        }

    def train(self, train_dataset: NeRFDataset, valid_dataset: NeRFDataset, max_epochs: int):

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_dataset.dataset.poses, train_dataset.dataset.intrinsics)

        self.error_map = train_dataset.dataset.error_map

        train_loader = train_dataset.dataloader()
        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch + 1
            self.train_one_epoch(data_loader=train_loader)

    def evaluate(self, test_dataset: NeRFDataset, name: str):
        pass

    def train_one_epoch(self, data_loader: torch.utils.data.DataLoader):
        self.model.train()

        total_loss = 0
        self.local_step = 0
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            self.global_step += 1
            self.local_step += 1
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % 200 == 0:
                with torch.amp.autocast('cuda', enabled=self.use_fp16):
                    self.model.update_extra_state()

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_fp16):
                preds, truths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.lr_scheduler.step()
            loss_val = loss.item()
            total_loss += loss_val

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        print(f"Epoch {self.epoch}, Loss: {average_loss:.4f}")

    def train_step(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        time = data['time']  # [B, 1]
        images = data['images']  # [B, N, 3/4]
        color_space = data['color_space']
        B, N, C = images.shape

        if color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
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

        outputs = self.model.render(
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

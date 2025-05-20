from .network import NeRFNetworkBasis
from .provider import NeRFDataset
import torch


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Trainer:
    def __init__(self,
                 name: str,
                 lr_encoding: float,
                 lr_net: float,
                 use_fp16: bool,
                 device
                 ):
        self.name = name
        self.model = NeRFNetworkBasis(
            encoding_spatial='tiledgrid',
            encoding_dir='sphere_harmonics',
            encoding_time='frequency',
            encoding_bg='hashgrid',
            bound=1,
        )
        self.optimizer = torch.optim.Adam(self.model.get_params(lr_encoding, lr_net), betas=(0.9, 0.99), eps=1e-15)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # TODO: implement a proper scheduler
        self.use_fp16 = use_fp16
        self.device = device

        self.epoch = 0
        self.global_step = 0
        self.to(self.device)

    def train(self, train_dataset: NeRFDataset, valid_dataset: NeRFDataset, max_epochs: int):

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_dataset.dataset.poses, train_dataset.dataset.intrinsics)

        train_loader = train_dataset.dataloader()
        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch + 1
            self.train_one_epoch(data_loader=train_loader)

    def evaluate(self, test_dataset: NeRFDataset, name: str):
        pass

    def train_one_epoch(self, data_loader: torch.utils.data.DataLoader):
        self.model.train()
        for i, data in enumerate(data_loader):
            self.global_step += 1
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % 200 == 0:
                with torch.amp.autocast('cuda', enabled=self.use_fp16):
                    self.model.update_extra_state()

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_fp16):
                # preds, truths, loss = self.train_step(data)
                self.train_step(data)

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

    def to(self, device):
        self.model.to(device)

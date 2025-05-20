from .network import NeRFNetworkBasis
from .provider import NeRFDataset
import torch


class Trainer:
    def __init__(self, name, lr_encoding, lr_net, device):
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
        self.device = device
        self.to(self.device)

        self.epoch = 0

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
            data: dict
            print(f"Processing batch {i + 1}/{data.keys()}")

            if self.model.cuda_ray:
                pass

    def to(self, device):
        self.model.to(device)

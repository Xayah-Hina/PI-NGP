from .network import NeRFNetworkBasis
import torch


class Trainer:
    def __init__(self, opt, device):
        self.name = opt.name
        self.model = NeRFNetworkBasis(
            encoding_spatial='tiledgrid',
            encoding_dir='sphere_harmonics',
            encoding_time='frequency',
            bound=1,
        )
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=5e-4)
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # TODO: implement a proper scheduler
        self.device = device
        self.to(self.device)

    def train(self, train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader, max_epochs: int):
        self.train_one_epoch(data_loader=train_loader)

    def evaluate(self, test_loader: torch.utils.data.DataLoader, name: str):
        pass

    def train_one_epoch(self, data_loader: torch.utils.data.DataLoader):
        for i, data in enumerate(data_loader):
            data: dict
            print(f"Processing batch {i + 1}/{data.keys()}")

    def to(self, device):
        self.model.to(device)

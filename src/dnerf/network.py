import torch


class NeRFRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, d, t):
        raise NotImplementedError()


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding,
                 ):
        super().__init__()

    def forward(self, x, d, t):
        pass

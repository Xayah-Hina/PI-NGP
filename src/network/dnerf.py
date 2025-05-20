from ..encoder import get_encoder
import torch
import typing


class NeRFRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, d, t):
        raise NotImplementedError()


class NeRFNetworkBasis(NeRFRenderer):
    def __init__(self,
                 encoding_spatial: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_dir: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_time: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 bound,
                 ):
        super().__init__()
        self.encoder_spatial = get_encoder(encoding_spatial, desired_resolution=2048 * bound)
        self.encoder_dir = get_encoder(encoding_dir)
        self.encoder_time = get_encoder(encoding_time, input_dim=1, multires=6)

    def forward(self, x, d, t):
        """
        :param x: [N, 3], in [-bound, bound]
        :param d: [N, 3], normalized in [-1, 1]
        :param t: [1, 1], in [0, 1]
        :return:
        """

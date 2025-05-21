from .renderer import NeRFRenderer
from ..encoder import get_encoder
from .activation import trunc_exp
import torch
import typing


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding_spatial: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_dir: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_bg: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 num_layers_sigma=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 ):
        super().__init__()

        # sigma network
        self.encoder_spatial = get_encoder(encoding_spatial, desired_resolution=2048 * bound)
        sigma_net = []
        for l in range(num_layers_sigma):
            if l == 0:
                in_dim = self.encoder_spatial.output_dim
            else:
                in_dim = hidden_dim
            if l == num_layers_sigma - 1:
                out_dim = 1 + geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = torch.nn.ModuleList(sigma_net)

        # color network
        self.encoder_dir = get_encoder(encoding_dir)
        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.encoder_dir.output_dim + geo_feat_dim
            else:
                in_dim = hidden_dim_color
            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim_color
            color_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
        self.color_net = torch.nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.encoder_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048)  # much smaller hashgrid
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.encoder_bg.output_dim + self.encoder_dir.output_dim
                else:
                    in_dim = hidden_dim_bg
                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                bg_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
            self.bg_net = torch.nn.ModuleList(bg_net)
        else:
            self.bg_net = None

        self.runtime_params = {
            'num_layers_sigma': num_layers_sigma,
            'num_layers_color': num_layers_color,
            'num_layers_bg': num_layers_bg,
            'bound': bound,
        }

    def forward(self, x, d):
        """
        :param x: [N, 3], in [-bound, bound]
        :param d: [N, 3], normalized in [-1, 1]
        :return: sigma, color
        """
        # sigma
        x = self.encoder_spatial(x, bound=self.runtime_params['bound'])
        h = x
        for l in range(self.runtime_params['num_layers_sigma']):
            h = self.sigma_net[l](h)
            if l != self.runtime_params['num_layers_sigma'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.runtime_params['num_layers_color']):
            h = self.color_net[l](h)
            if l != self.runtime_params['num_layers_color'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        """
        :param x: [N, 3], in [-bound, bound]
        :return:
        """
        x = self.encoder_spatial(x, bound=self.runtime_params['bound'])

        h = x
        for l in range(self.runtime_params['num_layers_sigma']):
            h = self.sigma_net[l](h)
            if l != self.runtime_params['num_layers_sigma'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def color(self, x, d, mask, geo_feat):
        """
        :param x: [N, 3], in [-bound, bound]
        :param d: [N, 3], normalized in [-1, 1]
        :param mask: [N,], bool, indicates where we actually needs to compute rgb.
        :param geo_feat: [N, geo_feat_dim], optional, if provided, we will not compute geo_feat again.
        :return:
        """
        raise DeprecationWarning("color() is not recommended")

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.runtime_params['num_layers_color']):
            h = self.color_net[l](h)
            if l != self.runtime_params['num_layers_color'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    def background(self, x, d):
        """
        :param x: [N, 2], in [-1, 1]
        :param d: [N, 3], normalized in [-1, 1]
        :return:
        """
        h = self.encoder_bg(x)  # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.runtime_params['num_layers_bg']):
            h = self.bg_net[l](h)
            if l != self.runtime_params['num_layers_bg'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

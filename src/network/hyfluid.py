from .renderer import NeRFRendererDynamic
from ..encoder import get_encoder
from .activation import trunc_exp
import torch
import typing


class NeRFHyFluidSmall(NeRFRendererDynamic):
    def __init__(self,
                 encoding_pinf: typing.Literal['hyfluid'],
                 num_layers_sigma=3,
                 hidden_dim_sigma=64,
                 ):
        super().__init__(
            cuda_ray=True,
        )

        self.encoder_pinf = get_encoder(encoding_pinf)
        self.rgb = torch.nn.Parameter(torch.tensor([0.0]))

        # sigma network
        sigma_net = []
        for l in range(num_layers_sigma):
            if l == 0:
                in_dim = self.encoder_pinf.num_levels * 2
            else:
                in_dim = hidden_dim_sigma
            if l == num_layers_sigma - 1:
                out_dim = 1  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim_sigma
            sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = torch.nn.ModuleList(sigma_net)

        self.runtime_params.update({
            'num_layers_sigma': num_layers_sigma,
        })

    def forward(self, x, d, t):
        xyzt = torch.cat([x, t.expand(x[..., :1].shape)], dim=-1)
        enc_xyzt = self.encoder_pinf(xyzt)

        h = enc_xyzt
        for l in range(self.runtime_params['num_layers_sigma']):
            h = self.sigma_net[l](h)
            h = torch.nn.functional.relu(h, inplace=True)

        sigma = h
        return sigma

    def density(self, x, t):
        results = {
            'sigma': self(x, None, t)
        }
        return results

    def color(self, x, d, t, mask=None, **kwargs):
        raise NotImplementedError("Color function is not implemented for NeRFHyFluidSmall.")

    def background(self, x, d):
        raise NotImplementedError("Background function is not implemented for NeRFHyFluidSmall.")

    def get_params(self, lr_encoding, lr_net):

        params = [
            {'params': self.encoder_pinf.parameters(), 'lr': lr_encoding},
            {'params': self.sigma_net.parameters(), 'lr': lr_net},
        ]

        return params


class NeRFSmall(torch.nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=2,
                 hidden_dim_color=16,
                 input_ch=3,
                 dtype=torch.float32,
                 ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.rgb = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False, dtype=dtype))

        self.sigma_net = torch.nn.ModuleList(sigma_net)

        self.color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = 1
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 1
            else:
                out_dim = hidden_dim_color

            self.color_net.append(torch.nn.Linear(in_dim, out_dim, bias=True, dtype=dtype))

    def forward(self, x):
        h = x[..., :self.input_ch]
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = torch.nn.functional.relu(h, inplace=True)

        sigma = h
        return sigma


class NeRFSmallPotential(torch.nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=2,
                 hidden_dim_color=16,
                 input_ch=3,
                 use_f=False,
                 dtype=torch.float32,
                 ):
        super(NeRFSmallPotential, self).__init__()

        self.input_ch = input_ch
        self.rgb = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = hidden_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False, dtype=dtype))
        self.sigma_net = torch.nn.ModuleList(sigma_net)
        self.out = torch.nn.Linear(hidden_dim, 3, bias=True, dtype=dtype)
        self.use_f = use_f
        if use_f:
            self.out_f = torch.nn.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype)
            self.out_f2 = torch.nn.Linear(hidden_dim, 3, bias=True, dtype=dtype)

    def forward(self, x):
        h = x[..., :self.input_ch]
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = torch.nn.functional.relu(h, True)

        v = self.out(h)
        if self.use_f:
            f = self.out_f(h)
            f = torch.nn.functional.relu(f, True)
            f = self.out_f2(f)
        else:
            f = v * 0
        return v, f


class NeRFSmall_c(torch.nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=2,
                 hidden_dim_color=16,
                 input_ch=3,
                 dtype=torch.float32,
                 ):
        super(NeRFSmall_c, self).__init__()

        self.input_ch = input_ch
        self.rgb = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(torch.nn.Linear(in_dim, out_dim, bias=False, dtype=dtype))

        self.sigma_net = torch.nn.ModuleList(sigma_net)

        self.color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = geo_feat_dim + 3
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3
            else:
                out_dim = hidden_dim_color

            self.color_net.append(torch.nn.Linear(in_dim, out_dim, bias=True, dtype=dtype))
        self.color_net = torch.nn.ModuleList(self.color_net)

    def forward(self, x):
        h = x[..., :self.input_ch]
        dirs = x[..., self.input_ch:]
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = torch.nn.functional.relu(h, inplace=True)

        sigma = h
        tmp = torch.cat([sigma[..., 1:], dirs], dim=-1)
        color = self.color_net[0](tmp)
        for l in range(1, self.num_layers_color):
            color = torch.nn.functional.relu(color, inplace=True)
            color = self.color_net[l](color)
        sigma_rgb = torch.cat([sigma[..., 0:1], color], dim=-1)
        return sigma_rgb

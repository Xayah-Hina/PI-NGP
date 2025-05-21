from .renderer import NeRFRendererDynamic
from ..encoder import get_encoder
from .activation import trunc_exp
import torch
import typing


class NeRFNetworkBasis(NeRFRendererDynamic):
    def __init__(self,
                 encoding_spatial: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_dir: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_time: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_bg: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 num_layers_sigma=2,
                 num_layers_color=3,
                 num_layers_basis=5,
                 num_layers_bg=2,
                 hidden_dim_sigma=64,
                 hidden_dim_color=64,
                 hidden_dim_basis=128,
                 hidden_dim_bg=64,
                 geo_feat_dim=32,
                 sigma_basis_dim=32,
                 color_basis_dim=8,
                 bound=1,
                 ):
        super().__init__(
            cuda_ray=True,
        )

        # basis network
        self.encoder_time = get_encoder(encoding_time, input_dim=1, multires=6)
        basis_net = []
        for l in range(num_layers_basis):
            if l == 0:
                in_dim = self.encoder_time.output_dim
            else:
                in_dim = hidden_dim_basis
            if l == num_layers_basis - 1:
                out_dim = sigma_basis_dim + color_basis_dim
            else:
                out_dim = hidden_dim_basis
            basis_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
        self.basis_net = torch.nn.ModuleList(basis_net)

        # sigma network
        self.encoder_spatial = get_encoder(encoding_spatial, desired_resolution=2048 * bound)
        sigma_net = []
        for l in range(num_layers_sigma):
            if l == 0:
                in_dim = self.encoder_spatial.output_dim
            else:
                in_dim = hidden_dim_sigma
            if l == num_layers_sigma - 1:
                out_dim = sigma_basis_dim + geo_feat_dim  # SB sigma + features for color
            else:
                out_dim = hidden_dim_sigma
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
                out_dim = 3 * color_basis_dim  # 3 * CB rgb
            else:
                out_dim = hidden_dim_color
            color_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
        self.color_net = torch.nn.ModuleList(color_net)

        # background network
        if self.runtime_params['bg_radius'] > 0:
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

        self.runtime_params.update({
            'num_layers_sigma': num_layers_sigma,
            'num_layers_basis': num_layers_basis,
            'num_layers_color': num_layers_color,
            'num_layers_bg': num_layers_bg,
            'sigma_basis_dim': sigma_basis_dim,
            'color_basis_dim': color_basis_dim,
            'bound': bound,
        })

    def forward(self, x, d, t):
        """
        :param x: [N, 3], in [-bound, bound]
        :param d: [N, 3], normalized in [-1, 1]
        :param t: [1, 1], in [0, 1]
        :return:
        """
        # time --> basis
        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        h = enc_t
        for l in range(self.runtime_params['num_layers_basis']):
            h = self.basis_net[l](h)
            if l != self.runtime_params['num_layers_basis'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)
        sigma_basis = h[0, :self.runtime_params['sigma_basis_dim']]
        color_basis = h[0, self.runtime_params['sigma_basis_dim']:]

        # sigma
        x = self.encoder_spatial(x, bound=self.runtime_params['bound'])
        h = x
        for l in range(self.runtime_params['num_layers_sigma']):
            h = self.sigma_net[l](h)
            if l != self.runtime_params['num_layers_sigma'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        sigma = trunc_exp(h[..., :self.runtime_params['sigma_basis_dim']] @ sigma_basis)
        geo_feat = h[..., self.runtime_params['sigma_basis_dim']:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.runtime_params['num_layers_color']):
            h = self.color_net[l](h)
            if l != self.runtime_params['num_layers_color'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h.view(-1, 3, self.runtime_params['color_basis_dim']) @ color_basis)

        return sigma, rgbs, None

    def density(self, x, t):
        """
        :param x: [N, 3], in [-bound, bound]
        :param t: [1, 1], in [0, 1]
        :return:
        """
        results = {}

        # time --> basis
        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        h = enc_t
        for l in range(self.runtime_params['num_layers_basis']):
            h = self.basis_net[l](h)
            if l != self.runtime_params['num_layers_basis'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        sigma_basis = h[0, :self.runtime_params['sigma_basis_dim']]
        color_basis = h[0, self.runtime_params['sigma_basis_dim']:]

        # sigma
        x = self.encoder_spatial(x, bound=self.runtime_params['bound'])
        h = x
        for l in range(self.runtime_params['num_layers_sigma']):
            h = self.sigma_net[l](h)
            if l != self.runtime_params['num_layers_sigma'] - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        sigma = trunc_exp(h[..., :self.runtime_params['sigma_basis_dim']] @ sigma_basis)
        geo_feat = h[..., self.runtime_params['sigma_basis_dim']:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat
        # results['color_basis'] = color_basis

        return results

    def color(self, x, d, t, mask=None, **kwargs):
        raise NotImplementedError('color is not implemented in NeRFNetworkBasis, please implement it in your own model.')

    def background(self, x, d):
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

    def get_params(self, lr_encoding, lr_net):

        params = [
            {'params': self.encoder_spatial.parameters(), 'lr': lr_encoding},
            {'params': self.sigma_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_dir.parameters(), 'lr': lr_encoding},
            {'params': self.color_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_time.parameters(), 'lr': lr_encoding},
            {'params': self.basis_net.parameters(), 'lr': lr_net},
        ]
        if self.runtime_params['bg_radius'] > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr_encoding})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})

        return params

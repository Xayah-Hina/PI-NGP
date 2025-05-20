from .activation import trunc_exp
from ..encoder import get_encoder
import torch
import typing
import math

from ..spatial import raymarching


class NeRFRenderer(torch.nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1,  # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()
        self.bound = bound
        self.bg_radius = bg_radius
        self.min_near = min_near
        self.density_scale = density_scale
        self.density_thresh = density_thresh

        ## parameters to be determined
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.time_size = 64
        self.grid_size = 128
        ## parameters to be determined

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        self.register_buffer('aabb_train', torch.tensor([-bound, -bound, -bound, bound, bound, bound], dtype=torch.float32))
        self.register_buffer('aabb_infer', torch.tensor([-bound, -bound, -bound, bound, bound, bound], dtype=torch.float32))

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid (with an extra time dimension)
            self.register_buffer('density_grid', torch.zeros(self.time_size, self.cascade, self.grid_size ** 3))  # [T, CAS, H * H * H]
            self.register_buffer('density_bitfield', torch.zeros(self.time_size, self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8))  # [T, CAS * H * H * H // 8]
            self.mean_density = 0
            self.iter_density = 0
            # time stamps for density grid
            self.register_buffer('times', ((torch.arange(self.time_size, dtype=torch.float32) + 0.5) / self.time_size).view(-1, 1, 1))  # [T, 1, 1]
            # step counter
            self.register_buffer('step_counter', torch.zeros(16, 2, dtype=torch.int32))  # 16 is hardcoded for averaging...
            self.mean_count = 0
            self.local_step = 0

    def forward(self, x, d, t):
        raise NotImplementedError('forward is not implemented in NeRFRenderer, please implement it in your own model.')

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x, t):
        raise NotImplementedError('density is not implemented in NeRFRenderer, please implement it in your own model.')

    def color(self, x, d, t, mask=None, **kwargs):
        raise NotImplementedError('color is not implemented in NeRFRenderer, please implement it in your own model.')

    def background(self, x, d):
        raise NotImplementedError('background is not implemented in NeRFRenderer, please implement it in your own model.')

    def reset_extra_state(self):
        if not self.cuda_ray:
            return
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.
        if not self.cuda_ray:
            return

        ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)

        # full update.
        if self.iter_density < 16:
            # if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for t, time in enumerate(self.times):
                for xs in X:
                    for ys in Y:
                        for zs in Z:

                            # construct points
                            xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [N, 3], in [0, 128)
                            indices = raymarching.morton3D(coords).long()  # [N]
                            xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                            # cascading
                            for cas in range(self.cascade):
                                bound = min(2 ** cas, self.bound)
                                half_grid_size = bound / self.grid_size
                                half_time_size = 0.5 / self.time_size
                                # scale to current cascade's resolution
                                cas_xyzs = xyzs * (bound - half_grid_size)
                                # add noise in coord [-hgs, hgs]
                                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                                # add noise in time [-hts, hts]
                                time_perturb = time + (torch.rand_like(time) * 2 - 1) * half_time_size
                                # query density
                                sigmas = self.density(cas_xyzs, time_perturb)['sigma'].reshape(-1).detach()
                                sigmas *= self.density_scale
                                # assign
                                tmp_grid[t, cas, indices] = sigmas

        # partial update (half the computation)
        # just update 100 times should be enough... too time consuming.
        elif self.iter_density < 100:
            N = self.grid_size ** 3 // 4  # T * C * H * H * H / 4
            for t, time in enumerate(self.times):
                for cas in range(self.cascade):
                    # random sample some positions
                    coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    # random sample occupied positions
                    occ_indices = torch.nonzero(self.density_grid[t, cas] > 0).squeeze(-1)  # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                    occ_indices = occ_indices[rand_mask]  # [Nz] --> [N], allow for duplication
                    occ_coords = raymarching.morton3D_invert(occ_indices)  # [N, 3]
                    # concat
                    indices = torch.cat([indices, occ_indices], dim=0)
                    coords = torch.cat([coords, occ_coords], dim=0)
                    # same below
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]
                    bound = min(2 ** cas, self.bound)
                    half_grid_size = bound / self.grid_size
                    half_time_size = 0.5 / self.time_size
                    # scale to current cascade's resolution
                    cas_xyzs = xyzs * (bound - half_grid_size)
                    # add noise in [-hgs, hgs]
                    cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                    # add noise in time [-hts, hts]
                    time_perturb = time + (torch.rand_like(time) * 2 - 1) * half_time_size
                    # query density
                    sigmas = self.density(cas_xyzs, time_perturb)['sigma'].reshape(-1).detach()
                    sigmas *= self.density_scale
                    # assign
                    tmp_grid[t, cas, indices] = sigmas

        ## max-pool on tmp_grid for less aggressive culling [No significant improvement...]
        # invalid_mask = tmp_grid < 0
        # tmp_grid = F.max_pool3d(tmp_grid.view(self.cascade, 1, self.grid_size, self.grid_size, self.grid_size), kernel_size=3, stride=1, padding=1).view(self.cascade, -1)
        # tmp_grid[invalid_mask] = -1

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item()  # -1 non-training regions are viewed as 0 density.
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        for t in range(self.time_size):
            raymarching.packbits(self.density_grid[t], density_thresh, self.density_bitfield[t])

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')

    def run_cuda(self, rays_o, rays_d, time, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024):
        """
        :param rays_o: [B, N, 3], assumes B == 1
        :param rays_d: [B, N, 3], assumes B == 1
        :param time: [B, 1]
        :param dt_gamma:
        :param bg_color:
        :param perturb:
        :param force_all_rays:
        :param max_steps:
        :return: image: [B, N, 3], depth: [B, N]
        """

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d)  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        # determine the correct frame of density grid to use
        t = torch.floor(time[0][0] * self.time_size).clamp(min=0, max=self.time_size - 1).long()

        results = {}

        if self.training:
            pass

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsics, S=64):
        """
        :param poses: [B, 4, 4]
        :param intrinsics: [4]
        :param S:
        :return:
        """

        if not self.cuda_ray:
            return

        B = poses.shape[0]
        fx, fy, cx, cy = [float(v) for v in intrinsics]

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid[0])
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0)  # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3]  # [S, N, 3]

                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0  # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1)  # [N]

                            # update count
                            count[cas, indices] += mask
                            head += S

        # mark untrained grid as -1
        self.density_grid[count.unsqueeze(0).expand_as(self.density_grid) == 0] = -1

        print(f'[mark untrained grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')


class NeRFNetworkBasis(NeRFRenderer):
    def __init__(self,
                 encoding_spatial: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_dir: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_time: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 encoding_bg: typing.Literal['None', 'frequency', 'sphere_harmonics', 'hashgrid', 'tiledgrid', 'ash'],
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=32,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 sigma_basis_dim=32,
                 color_basis_dim=8,
                 num_layers_basis=5,
                 hidden_dim_basis=128,
                 bound=1,
                 ):
        super().__init__(
            cuda_ray=True,
            bound=bound,
            bg_radius=-1,
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
        for l in range(num_layers):
            if l == 0:
                in_dim = self.encoder_spatial.output_dim
            else:
                in_dim = hidden_dim
            if l == num_layers - 1:
                out_dim = sigma_basis_dim + geo_feat_dim  # SB sigma + features for color
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
                out_dim = 3 * color_basis_dim  # 3 * CB rgb
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

        self.params = {
            'num_layers': num_layers,
            'num_layers_basis': num_layers_basis,
            'num_layers_color': num_layers_color,
            'num_layers_bg': num_layers_bg,
            'sigma_basis_dim': sigma_basis_dim,
            'color_basis_dim': color_basis_dim,
            'bound': bound,
        }

    def forward(self, x, d, t):
        """
        :param x: [N, 3], in [-bound, bound]
        :param d: [N, 3], normalized in [-1, 1]
        :param t: [1, 1], in [0, 1]
        :return:
        """
        num_layers = self.params['num_layers']
        num_layers_basis = self.params['num_layers_basis']
        num_layers_color = self.params['num_layers_color']
        sigma_basis_dim = self.params['sigma_basis_dim']
        color_basis_dim = self.params['color_basis_dim']
        bound = self.params['bound']

        # time --> basis
        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        h = enc_t
        for l in range(num_layers_basis):
            h = self.basis_net[l](h)
            if l != num_layers_basis - 1:
                h = torch.nn.functional.relu(h, inplace=True)
        sigma_basis = h[0, :sigma_basis_dim]
        color_basis = h[0, sigma_basis_dim:]

        # sigma
        x = self.encoder(x, bound=bound)
        h = x
        for l in range(num_layers):
            h = self.sigma_net[l](h)
            if l != num_layers - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        sigma = trunc_exp(h[..., :sigma_basis_dim] @ sigma_basis)
        geo_feat = h[..., sigma_basis_dim:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(num_layers_color):
            h = self.color_net[l](h)
            if l != num_layers_color - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h.view(-1, 3, color_basis_dim) @ color_basis)

        return sigma, rgbs, None

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]

        num_layers = self.params['num_layers']
        num_layers_basis = self.params['num_layers_basis']
        sigma_basis_dim = self.params['sigma_basis_dim']

        results = {}

        # time --> basis
        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        h = enc_t
        for l in range(num_layers_basis):
            h = self.basis_net[l](h)
            if l != num_layers_basis - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        sigma_basis = h[0, :sigma_basis_dim]
        color_basis = h[0, sigma_basis_dim:]

        # sigma
        x = self.encoder_spatial(x, bound=self.bound)
        h = x
        for l in range(num_layers):
            h = self.sigma_net[l](h)
            if l != num_layers - 1:
                h = torch.nn.functional.relu(h, inplace=True)

        sigma = trunc_exp(h[..., :sigma_basis_dim] @ sigma_basis)
        geo_feat = h[..., sigma_basis_dim:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat
        # results['color_basis'] = color_basis

        return results

    def color(self, x, d, t, mask=None, **kwargs):
        raise NotImplementedError('color is not implemented in NeRFNetworkBasis, please implement it in your own model.')

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        num_layers_bg = self.params['num_layers_bg']

        h = self.encoder_bg(x)  # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(num_layers_bg):
            h = self.bg_net[l](h)
            if l != num_layers_bg - 1:
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
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr_encoding})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})

        return params

from ..spatial import raymarching
import torch
import math


class NeRFRenderer(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask, geo_feat):
        raise NotImplementedError()

    def background(self, x, d):
        raise NotImplementedError('background is not implemented in NeRFRenderer, please implement it in your own model.')


class NeRFRendererDynamic(torch.nn.Module):
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

        print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128 ** 3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')

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
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()  # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield[t], self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

            # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

            sigmas, rgbs, deform = self(xyzs, dirs, time)
            # density_outputs = self.density(xyzs, time) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
            # sigmas = density_outputs['sigma']
            # rgbs = self.color(xyzs, dirs, **density_outputs)
            sigmas = self.density_scale * sigmas

            # print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)

            results['deform'] = deform
        else:
            # allocate outputs
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            # dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)  # [N]
            rays_t = nears.clone()  # [N]

            step = 0

            while step < max_steps:

                # count alive rays
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield[t], self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs, _ = self(xyzs, dirs, time)
                # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
                # sigmas = density_outputs['sigma']
                # rgbs = self.color(xyzs, dirs, **density_outputs)
                sigmas = self.density_scale * sigmas

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image)

                rays_alive = rays_alive[rays_alive >= 0]

                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)

        results['depth'] = depth
        results['image'] = image
        return results

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

    def render(self, rays_o, rays_d, time, dt_gamma, bg_color, perturb, force_all_rays, max_steps):
        if self.cuda_ray:
            results = self.run_cuda(
                rays_o=rays_o,
                rays_d=rays_d,
                time=time,
                dt_gamma=dt_gamma,
                bg_color=bg_color,
                perturb=perturb,
                force_all_rays=force_all_rays,
                max_steps=max_steps,
            )
        else:
            raise NotImplementedError('NeRFRenderer.render() is not implemented for non-cuda-ray mode.')
        return results

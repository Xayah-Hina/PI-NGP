from .freqencoder import FreqEncoder
from .gridencoder import GridEncoder
from .shencoder import SHEncoder


def get_encoder(encoding,
                input_dim=3,
                multires=6,
                degree=4,
                num_levels=16,
                level_dim=2,
                base_resolution=16,
                log2_hashmap_size=19,
                desired_resolution=2048,
                align_corners=False,
                ):
    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    elif encoding == 'frequency':
        # encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'sphere_harmonics':
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)

    elif encoding == 'tiledgrid':
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder

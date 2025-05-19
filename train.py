from src import *
import torch
import os
import argparse


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=os.path.abspath(os.path.join(os.getcwd(), 'data')), help='data directory')  # used by [NeRFDataset]
    parser.add_argument('--dataset_dir', type=str, default='dnerf/standup', help='data directory')  # used by [NeRFDataset]
    parser.add_argument('--camera_radius_scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")  # used by [NeRFDataset]
    parser.add_argument('--camera_offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")  # used by [NeRFDataset]
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")  # used by [NeRFDataset]
    parser.add_argument('--color_space', type=str, default='srgb', choices=['linear', 'srgb'], help="Color space, supports (linear, srgb)")  # used by [NeRFDataset]
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")  # used by [NeRFDataset]
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")  # used by [NeRFDataset]
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")  # used by [NeRFDataset]
    parser.add_argument('--num_rays', type=int, default=4096, help="batch rays")  # used by [NeRFDataset]

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_argument()
    opt.name = 'PI-NGP'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(opt=opt, device=device)
    trainer.train(
        train_loader=NeRFDataset(opt, dataset_type='train', num_rays=getattr(opt, 'num_rays'), device=device).dataloader(),
        valid_loader=NeRFDataset(opt, dataset_type='val', num_rays=getattr(opt, 'num_rays'), device=device).dataloader(),
        max_epochs=20,
    )
    trainer.evaluate(
        test_loader=NeRFDataset(opt, dataset_type='test', num_rays=getattr(opt, 'num_rays'), device=device).dataloader(),
        name="test",
    )

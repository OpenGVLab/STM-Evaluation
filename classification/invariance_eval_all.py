# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import argparse
from pathlib import Path
from xml.sax import default_parser_list

import torch
import numpy as np
from torch.backends import cudnn
from timm.models import create_model
from datasets import build_dataset
from engine import evaluate_invariance, VarianceCollateFN


# timm register while not used
import utils
import models
from main import str2bool
from tools.variance_transforms import standard_transform, position_jitter_transform, rotate_transform


def get_args_parser():
    parser = argparse.ArgumentParser('evaluation script for variance in image classification',
                                     add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Per GPU batch size')

    # Model parameters
    parser.add_argument('--model', default='conv_swin_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', type=float, default=1e-6,
                        help='the initial value for layer scale, default=1e-6')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=0.875)

    # Dataset parameters
    parser.add_argument('--data_path', default='./minidata', type=str,
                        help='dataset path')
    parser.add_argument('--data_on_memory', default=False, type=str2bool,
                        help='loading training data to memory')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--variance_type', default="translation",
                        choices=['translation', 'pre_rotation', 'post_rotation', 'scale'])
    # parser.add_argument('--jitter_strength', default=0, type=int)
    # parser.add_argument('--rotation_angle', default=0, type=int)

    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET1k',
                        # choices=['CIFAR', 'IMNET', 'image_folder'],
                        choices=['IMNET1k'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='/data1/shimin/model_parameters/backbone/swin_tiny/checkpoint-best.pth',
                        help='resume from checkpoint')

    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='unified_model_eval', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--name', default='unified_model_eval', type=str,
                        help="The name of the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    dataset_val.transform = None

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            # print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
            #       'This will slightly alter validation results as extra duplicate entries are added to achieve '
            #       'equal num of samples per-process.', file=sys.stderr)
            raise NotImplementedError('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                                      'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(dataset_val,
                                                          num_replicas=num_tasks,
                                                          rank=global_rank,
                                                          shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    transform = standard_transform(img_size=args.input_size,
                                   crop_ratio=args.crop_pct)

    variance_transforms = {}

    if args.variance_type == 'translation':
        for strength in range(0, 64+1, 4):
            variance_transforms[f'position jitter {strength}'] = position_jitter_transform(img_size=args.input_size,
                                                                                           crop_ratio=args.crop_pct,
                                                                                           jitter_strength=strength)

    if args.variance_type == 'pre_rotation':
        for angle in range(0, 90+1, 5):
            variance_transforms[f'pre rotation {angle}'] = rotate_transform(img_size=args.input_size,
                                                                            crop_ratio=args.crop_pct,
                                                                            angle=angle,
                                                                            pre_rotate=True)

    if args.variance_type == 'post_rotation':
        for angle in range(0, 90+1, 5):
            variance_transforms[f'pre rotation {angle}'] = rotate_transform(img_size=args.input_size,
                                                                            crop_ratio=args.crop_pct,
                                                                            angle=angle,
                                                                            pre_rotate=False)

    if args.variance_type == 'scale':
        for ratio in range(200, 2000+1, 125):
            ratio = ratio / 1000
            variance_transforms[f'scale {ratio}'] = standard_transform(img_size=args.input_size,
                                                                       crop_ratio=ratio)

    print('num variance transforms:', len(variance_transforms))

    collate_fn = VarianceCollateFN(standard_transform=transform,
                                   variance_transforms=variance_transforms)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        layer_scale_init_value=args.layer_scale_init_value,
    ).to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = {args.model}")
    print('number of params:', n_parameters)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model = model.cuda()
        model_without_ddp = model

    args.auto_resume = None
    utils.auto_load_model(args=args,
                          model=model,
                          model_without_ddp=model_without_ddp,
                          optimizer=None, loss_scaler=None, model_ema=None)

    test_stats = evaluate_invariance(data_loader_val, model, device, use_amp=args.use_amp)

    with open(os.path.join(args.output_dir, f'variance_{args.variance_type}.txt'), 'w', encoding='utf-8') as file:
        file.write('* Eval Results\n')
        for key, value in test_stats.items():
            file.write(f'\t{key}: {value}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    # srun -p VC --gres=gpu:1 --quotatype=spot --ntasks=1 --ntasks-per-node=1 python invariance_eval_all.py --batch_size 1024 --data_path /mnt/cache/share/images/ --output_dir ./backbone_outputdir/eval --model conv_swin_tiny --resume /mnt/petrelfs/share_data/shimin/share_checkpoint/swin/swin_tiny/checkpoint-best.pth

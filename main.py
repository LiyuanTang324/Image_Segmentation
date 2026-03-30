import argparse
import os
import random
import numpy as np
import torch
from torch.backends import cudnn
from solver import Solver
from data_loader import get_loader

DATASETS = ['BUS-BRA', 'BUSI', 'BUSIS', 'CAMUS', 'DDTI', 'Fetal_HC', 'KidneyUS']
MODELS = ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config):
    set_seed(config.seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

    image_dir = os.path.join(config.image_root, config.dataset, 'imgs')
    mask_dir = os.path.join(config.image_root, config.dataset, 'masks')
    split_dir = os.path.join(config.split_root, config.dataset)

    model_path = os.path.join(
        config.output_root, config.dataset, config.model_type, 'models')
    result_path = os.path.join(
        config.output_root, config.dataset, config.model_type, 'results')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    config.model_path = model_path
    config.result_path = result_path

    print(f'\n{"=" * 55}')
    print(f'  {config.model_type}  on  {config.dataset}  [{config.mode}]')
    print(f'{"=" * 55}')
    print(f'  Images : {image_dir}')
    print(f'  Splits : {split_dir}')
    print(f'  Output : {os.path.dirname(model_path)}')
    print()

    train_loader = get_loader(
        image_dir, mask_dir, os.path.join(split_dir, 'train.txt'),
        image_size=config.image_size, batch_size=config.batch_size,
        num_workers=config.num_workers, mode='train',
        augmentation_prob=config.augmentation_prob)

    val_loader = get_loader(
        image_dir, mask_dir, os.path.join(split_dir, 'val.txt'),
        image_size=config.image_size, batch_size=config.batch_size,
        num_workers=config.num_workers, mode='val')

    test_loader = get_loader(
        image_dir, mask_dir, os.path.join(split_dir, 'test.txt'),
        image_size=config.image_size, batch_size=1,
        num_workers=config.num_workers, mode='test')

    solver = Solver(config, train_loader, val_loader, test_loader)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image Segmentation – Control Experiment Group A')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASETS,
                        help='Target dataset name')
    parser.add_argument('--model_type', type=str, default='U_Net',
                        choices=MODELS,
                        help='Model architecture')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='train (includes test at end) or test only')

    parser.add_argument('--image_root', type=str,
                        default='/root/autodl-tmp/XpertUS/data/segmentation',
                        help='Root dir containing <dataset>/imgs/ and masks/')
    parser.add_argument('--split_root', type=str,
                        default='/root/autodl-tmp/control_experiment/data/segmentation',
                        help='Root dir containing <dataset>/train.txt etc.')
    parser.add_argument('--output_root', type=str, default='./output',
                        help='Root dir for checkpoints and results')

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--t', type=int, default=2,
                        help='Recurrent steps for R2U_Net / R2AttU_Net')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    config = parser.parse_args()
    main(config)

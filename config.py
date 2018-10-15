#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser()

parser.add_argument('--img_H', type=int, default=256,
                     help='input image height')
parser.add_argument('--img_W', type=int, default=256,
                     help='input image width')
parser.add_argument('--repeat_num', type=int, default=6,
                     help='number of blocks in the paper')
parser.add_argument('--conv_hidden_num', type=int, default=128,
                     choices=[64, 128], help='n in the paper')
parser.add_argument('--z_num', type=int, default=64)

parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--start_step', type=int, default=0)
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--checkpoint_dir', type=str, default='./data')
parser.add_argument('--max_step', type=int, default=500000)
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--d_lr', type=float, default=0.00008)
parser.add_argument('--g_lr', type=float, default=0.00008)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lambda_k', type=float, default=0.001)
parser.add_argument('--use_gpu', type=str2bool, default=True)
parser.add_argument('--gpu', type=int, default=-1)

parser.add_argument('--data_dir', type=str, default='./data')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

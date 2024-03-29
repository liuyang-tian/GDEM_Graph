import optuna

import torch
import argparse
import random
import numpy as np
from utils import *
from graph_agent import GraphAgent
import math


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save', type=int, default=1, help='whether to save the condensed graphs')

parser.add_argument('--dataset', type=str, default='ogbg-molbbbp')
parser.add_argument('--ipc', type=int, default=10, help='number of condensed samples per class')

parser.add_argument('--init', type=str, default='real')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--ratio_k', type=float, default=0.2)
parser.add_argument('--e1', type=int, default=1)
parser.add_argument('--e2', type=int, default=5)
parser.add_argument('--alpha', type=float, default=10.0)
parser.add_argument('--beta', type=float, default=0.00001)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--lr_feat', type=float, default=0.001)
parser.add_argument('--lr_eigenvec', type=float, default=0.001)
parser.add_argument('--bs_cond', type=int, default=256, help='batch size for sampling graphs')
parser.add_argument('--model_epoch', type=int, default=500)
parser.add_argument('--nconvs', type=int, default=3)
parser.add_argument('--lr_model', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--pooling', type=str, default='sum')
parser.add_argument('--hidden', type=int, default=128)

parser.add_argument('--net_norm', type=str, default='none')
parser.add_argument('--reduction_rate', type=float, default=0.1, help='if ipc=0, this param  will be enabled')
parser.add_argument('--stru_discrete', type=int, default=1)

args = parser.parse_args()

if args.dataset == 'ogbg-molhiv':
    args.pooling = 'sum'
if args.dataset == 'CIFAR10':
    args.net_norm = 'instancenorm'
if args.dataset == 'MUTAG' and args.ipc == 50:
    args.ipc = 20
    
torch.cuda.set_device(args.gpu_id)

print(args)
device = f'cuda:{args.gpu_id}'

args.k1 = math.ceil(args.K * args.ratio_k)
args.k2 = args.K - args.k1
print("k1:", args.k1, ",", "k2:", args.k2)

data = Dataset(args)
packed_data = data.packed_data

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

agent = GraphAgent(data=packed_data, args=args, device=device, nnodes_syn=get_mean_nodes(args))
mean_acc = agent.train()
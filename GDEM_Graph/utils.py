from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DenseDataLoader
import os.path as osp
from torch_geometric.datasets import MNISTSuperpixels
import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import random
import tqdm
import scipy.sparse as sp


class Complete(object):
    def __call__(self, data):
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)
        return data
    
    
class ConcatPos(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        data.x = torch.cat([data.x, data.pos], dim=1)
        data.pos = None
        return data
  

class RemoveEdgeAttr(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)

        data.y = data.y.squeeze(0)
        data.x = data.x.float()
        return data
  
    
class Dataset:

    def __init__(self, args):
        # random seed setting
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        name = args.dataset
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', f'{name}')

        if name in ['DD', 'MUTAG', 'NCI1']:
            dataset = TUDataset(path, name=name, transform=T.Compose([Complete()]), use_node_attr=True)
            dataset = dataset.shuffle()
            n = (len(dataset) + 9) // 10
            test_dataset = dataset[:n]
            val_dataset = dataset[n:2 * n]
            train_dataset = dataset[2 * n:]
            nnodes = [x.num_nodes for x in dataset]
            print('mean #nodes:', np.mean(nnodes), 'max #nodes:', np.max(nnodes), 'min #nodes:', np.min(nnodes))

        if name in ['CIFAR10']:
            transform = T.Compose([ConcatPos()])
            train_dataset= GNNBenchmarkDataset(path, name=name, split='train', transform=transform)
            val_dataset= GNNBenchmarkDataset(path, name=name, split='val', transform=transform)
            test_dataset= GNNBenchmarkDataset(path, name=name, split='test', transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
            nnodes = [x.num_nodes for x in train_dataset]
            print('mean #nodes:', np.mean(nnodes), 'max #nodes:', np.max(nnodes))


        if name in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
            dataset = PygGraphPropPredDataset(name=name, transform=T.Compose([RemoveEdgeAttr()]))
            split_idx = dataset.get_idx_split()
            train_dataset = dataset[split_idx["train"]]
            nnodes = [x.num_nodes for x in train_dataset]
            print('mean #nodes:', np.mean(nnodes), 'max #nodes:', np.max(nnodes))
            ### automatic evaluator. takes dataset name as input
            train_dataset = dataset[split_idx["train"]]
            val_dataset = dataset[split_idx["valid"]]
            test_dataset = dataset[split_idx["test"]]


        y_final = [g.y.item() for g in test_dataset]
        from collections import Counter; counter=Counter(y_final); print(counter)
        print("#Majority guessing:", sorted(counter.items())[-1][1]/len(y_final))

        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        train_datalist = np.ndarray((len(train_dataset),), dtype=np.object_)
        for ii in range(len(train_dataset)):
            train_datalist[ii] = train_dataset[ii]
            
            num_node = train_dataset[ii].x.size(0)            
            A_ = get_adj_norm(train_dataset[ii].edge_index, num_node)
            if num_node >= args.K:
                e, u = torch.linalg.eigh(A_)
                train_datalist[ii].e = torch.cat([e[:args.k1], e[(num_node-args.k2):]]).unsqueeze(0)
                train_datalist[ii].u = torch.cat([u[:,:args.k1], u[:,(num_node-args.k2):]], dim=1)
            else:
                train_datalist[ii].e = 0.
                train_datalist[ii].u = None

        self.packed_data = [train_dataset, train_loader, val_loader, test_loader, train_datalist]

def get_mean_nodes(args):
    if args.dataset == 'CIFAR10':
        return 118
    if args.dataset == 'DD':
        return 285
    if args.dataset == 'MUTAG':
        return 18
    if args.dataset == 'NCI1':
        return 30
    if args.dataset == 'ogbg-molhiv':
        return 26
    if args.dataset == 'ogbg-molbbbp':
        return 24
    if args.dataset == 'ogbg-molbace':
        return 34

    raise NotImplementedError


def get_min_nodes(args):
    if args.dataset == 'DD':
        return 30
    if args.dataset == 'ogbg-molbace':
        return 10
    raise NotImplementedError

def get_adj_norm(edge_index, num_nodes):
    src, dst = edge_index
    num_nodes = num_nodes

    if num_nodes == 1:  # some graphs have one node
        A_ = torch.tensor(1.).view(1, 1)
    else:
        A = torch.zeros([num_nodes, num_nodes], dtype=torch.float)
        A[src, dst] = 1.0
        for i in range(num_nodes):
            A[i, i] = 1.0
        deg = torch.sum(A, axis=0).squeeze() ** -0.5
        D = torch.diag(deg)
        A_ = D @ A @ D
    
    return A_


def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

class SparseTensorDataset(Dataset):
    def __init__(self, data): # images: n x c x h x w tensor
        self.data  = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
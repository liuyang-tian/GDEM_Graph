import os.path as osp
import os
from math import ceil
import torch
import torch.nn.functional as F
from models_gcn import GCN
from models import DenseGCN
# from dense_sgc import get_akx
from collections import Counter
import numpy as np
from utils import *
from copy import deepcopy
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score

import networkx as nx
import math

cls_criterion = torch.nn.BCEWithLogitsLoss()

class GraphAgent:

    def __init__(self, data, args, device, nnodes_syn=75):
        self.data = data
        self.args = args
        self.device = device
        labels_train = [x.y.item() for x in data[0]]

        self.k1 = args.k1
        self.k2 = args.k2
        
        print('training size:', len(labels_train))
        nfeat = data[0].num_features
        nclass = data[0].num_classes

        self.prepare_train_indices()

        # parametrize syn data
        self.labels_syn = self.get_labels_syn(labels_train)
        if args.ipc == 0:
            n = int(len(labels_train) * args.reduction_rate)
        else:
            self.labels_syn = torch.LongTensor([[i]*args.ipc for i in range(nclass)]).to(device).view(-1)
            self.syn_class_indices = {i: [i*args.ipc, (i+1)*args.ipc] for i in range(nclass)}
            n = args.ipc * nclass
            
        self.train_list = []
        for i_c in range(nclass):
            num_c_samples = len(self.real_indices_class[i_c])
            num_parts = args.ipc
            initial_step = num_c_samples//args.ipc if num_c_samples%args.ipc==0 else num_c_samples//args.ipc+1
            indices = [i * initial_step for i in range(num_parts)]
            indices.append(num_c_samples)
            self.train_list.append(indices)
            
        self.eigenvec_syn = torch.rand(size=(n, nnodes_syn, args.K), dtype=torch.float, requires_grad=True, device=device)
        self.feat_syn = torch.rand(size=(n, nnodes_syn, nfeat), dtype=torch.float, requires_grad=True, device=device)
        
        if args.init == 'real':
            eigenvec_init_list = []
            for c in range(nclass):
                for ipc_c in range(args.ipc):
                    ind = self.syn_class_indices[c]
                    feat_real, _, _, _ = self.get_graphs(ipc_c, c, batch_size=1, max_node_size=nnodes_syn, to_dense=True)
                    self.feat_syn.data[ind[0]: ind[1]][ipc_c] = feat_real[:, :nnodes_syn].detach().data

                    ind_s = self.train_list[c][ipc_c]
                    ind_e = self.train_list[c][ipc_c+1]
                    indices = np.array(self.real_indices_class[c][ind_s:ind_e])[self.nnodes_all[self.real_indices_class[c][ind_s:ind_e]] == nnodes_syn]
                    
                    if indices.shape[0]==0:
                        # 生成一个图然后分解
                        eigenvec_init=self.get_init_syn_eigenvecs(nnodes_syn, num_classes=1)
                        eigenvec_init_list.append(torch.cat([eigenvec_init[:,:self.k1], eigenvec_init[:,(nnodes_syn-self.k2):]], dim=1))
                    else:
                        idx_shuffle = np.random.permutation(indices)[0]
                        sampled = self.data[4][idx_shuffle]
                        eigenvec_init_list.append(torch.FloatTensor(sampled.u).cuda())
                                            
            eigenvec_init_ = torch.stack(eigenvec_init_list, dim=0)
            self.eigenvec_syn.data = eigenvec_init_.data  
        else:
            print("***************************")

        print('feat.shape:', self.feat_syn.shape)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_eigenvec = torch.optim.Adam([self.eigenvec_syn], lr=args.lr_eigenvec)
        
        self.weights = []
        self.eigenvals_list = []
        self.eigenvals_syn = None
        

    def prepare_train_indices(self):
        dataset = self.data[0]
        indices_class = {}
        nnodes_all = []
        for ix, single in enumerate(dataset):
            c = single.y.item()
            if c not in indices_class:
                indices_class[c] = [ix]
            else:
                indices_class[c].append(ix)
            nnodes_all.append(single.num_nodes)

        self.nnodes_all = np.array(nnodes_all)
        self.real_indices_class = indices_class

    def get_labels_syn(self, labels_train):
        counter = Counter(labels_train)
        num_class_dict = {}
        n = len(labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}

        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return torch.LongTensor(labels_syn).to(self.device)

    def get_graphs(self, ipc_c, c, batch_size, max_node_size=None, to_dense=False, idx_selected=None):
        """get random n images from class c"""
        ind_s = self.train_list[c][ipc_c]
        ind_e = self.train_list[c][ipc_c+1]

        if idx_selected is None:
            if max_node_size is None:
                idx_shuffle = np.random.permutation(self.real_indices_class[c][ind_s:ind_e])[:batch_size]
                sampled = self.data[4][idx_shuffle]
            else:
                indices = np.array(self.real_indices_class[c][ind_s:ind_e])[self.nnodes_all[self.real_indices_class[c][ind_s:ind_e]] <= max_node_size]
                idx_shuffle = np.random.permutation(indices)[:batch_size]
                sampled = self.data[4][idx_shuffle]
        else:
            sampled = self.data[4][idx_selected]
        data = Batch.from_data_list(sampled)
        if to_dense:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            e = data.e
            u = data.u
            x, mask = to_dense_batch(x, batch=batch, max_num_nodes=max_node_size)
            e, _ = to_dense_batch(e, batch=torch.arange(data.e.shape[0]))
            u, _ = to_dense_batch(u, batch=batch, max_num_nodes=max_node_size)
            return x.to(self.device), e.cuda(), u.cuda(), data.batch.cuda()
        
        else:
            return data.to(self.device)
        
    def get_graphs_min(self, ipc_c, c, batch_size, min_node_size=None, to_dense=False, idx_selected=None):
            """get random n images from class c"""
            ind_s = self.train_list[c][ipc_c]
            ind_e = self.train_list[c][ipc_c+1]

            if idx_selected is None:
                if min_node_size is None:
                    idx_shuffle = np.random.permutation(self.real_indices_class[c][ind_s:ind_e])[:batch_size]
                    sampled = self.data[4][idx_shuffle]
                else:
                    indices = np.array(self.real_indices_class[c][ind_s:ind_e])[self.nnodes_all[self.real_indices_class[c][ind_s:ind_e]] >= min_node_size]
                    idx_shuffle = np.random.permutation(indices)[:batch_size]
                    sampled = self.data[4][idx_shuffle]
            else:
                sampled = self.data[4][idx_selected]
            data = Batch.from_data_list(sampled)
            if to_dense:
                x, edge_index, batch = data.x, data.edge_index, data.batch
                e = data.e
                u = data.u
                x, mask = to_dense_batch(x, batch=batch)
                e, _ = to_dense_batch(e, batch=torch.arange(data.e.shape[0]))
                u, _ = to_dense_batch(u, batch=batch)
                return x.to(self.device), e.cuda(), u.cuda(), data.batch.cuda()
            
            else:
                return data.to(self.device)

    def get_graphs_multiclass(self, batch_size, max_node_size=None, idx_herding=None):
        """get random n graphs from classes"""
        if idx_herding is None:
            if max_node_size is None:
                idx_shuffle = []
                for c in range(self.data[0].num_classes):
                    idx_shuffle.append(np.random.permutation(self.real_indices_class[c])[:batch_size])
                idx_shuffle = np.hstack(idx_shuffle)
                sampled = self.data[4][idx_shuffle]
            else:
                idx_shuffle = []
                for c in range(self.data[0].num_classes):
                    indices = np.array(self.real_indices_class[c])[self.nnodes_all[self.real_indices_class[c]] <= max_node_size]
                    idx_shuffle.append(np.random.permutation(indices)[:batch_size])
                idx_shuffle = np.hstack(idx_shuffle)
                sampled = self.data[4][idx_shuffle]
        else:
            sampled = self.data[4][idx_herding]
        data = Batch.from_data_list(sampled)
        return data.to(self.device)

    def clip(self):
        self.adj_syn.data.clamp_(min=0, max=1)
        # self.feat_syn.data.clamp_(min=0, max=1)

    def get_init_syn_eigenvecs(self, n_syn, num_classes=1):
            n_nodes_per_class = n_syn // num_classes
            n_nodes_last = n_syn % num_classes

            size = [n_nodes_per_class for i in range(num_classes - 1)] + (
                [n_syn - (num_classes - 1) * n_nodes_per_class] if n_nodes_last != 0 else [n_nodes_per_class]
            )
            prob_same_community = 1 / num_classes
            prob_diff_community = prob_same_community / 3

            prob = [
                [prob_diff_community for i in range(num_classes)]
                for i in range(num_classes)
            ]
            for idx in range(num_classes):
                prob[idx][idx] = prob_same_community

            syn_graph = nx.stochastic_block_model(size, prob)
            syn_graph_adj = nx.adjacency_matrix(syn_graph)
            syn_graph_adj_norm = normalize_adj(syn_graph_adj)
            # syn_graph_L = np.eye(n_syn) - syn_graph_L
            _, eigen_vecs = np.linalg.eigh(syn_graph_adj_norm.todense())

            return torch.FloatTensor(eigen_vecs).cuda()


    def test(self, epochs=500, save=False, verbose=False, new_data=None):
        dataset = self.data[0]
        args = self.args
                
        if new_data is None:
            feat_syn = self.feat_syn.detach() # b,N',d
            eigenvecs_syn = self.eigenvec_syn.detach() # b,N',K
            eigenvals_syn = self.eigenvals_syn # b,K
            adj_syn = eigenvecs_syn @ torch.diag_embed(eigenvals_syn) @ eigenvecs_syn.permute(0,2,1) # b,N'N'

        labels_syn = self.labels_syn

        return self.test_pyg_data(feat_syn, adj_syn, labels_syn, epochs=epochs)

    def test_pyg_data(self, feat_syn, adj_syn, labels_syn, epochs=500, save=False, verbose=False):
        dataset = self.data[0]
        args = self.args
        use_val = True
        model_syn = DenseGCN(nfeat=dataset.num_features, nhid=args.hidden, dropout=args.dropout, net_norm=args.net_norm,
                        nconvs=args.nconvs, nclass=dataset.num_classes, pooling=args.pooling, args=args).cuda()
        model_real = GCN(nfeat=dataset.num_features, dropout=0.0, net_norm=args.net_norm,
                nconvs=args.nconvs, nhid=args.hidden, nclass=dataset.num_classes, pooling=args.pooling, args=args).cuda()
        
        lr = args.lr_model
        optimizer = torch.optim.Adam(model_syn.parameters(), lr=lr)

        @torch.no_grad()
        def test(loader, report_metric=False):
            model_real.eval()
            if self.args.dataset in ['ogbg-molhiv','ogbg-molbbbp', 'ogbg-molbace']:
                pred, y = [], []
                for data in loader:
                    data = data.to(self.device)
                    pred.append(model_real(data))
                    y.append(data.y.view(-1,1))
                from ogb.graphproppred import Evaluator;
                evaluator = Evaluator(self.args.dataset)
                return evaluator.eval({'y_pred': torch.cat(pred),
                            'y_true': torch.cat(y)})['rocauc']
            else:
                correct = 0
                for data in loader:
                    data = data.to(self.device)
                    pred = model_real(data).max(dim=1)[1]
                    correct += pred.eq(data.y.view(-1)).sum().item()
                    if report_metric:
                        nnodes_list = [(data.ptr[i]-data.ptr[i-1]).item() for i in range(1, len(data.ptr))]
                        low = np.quantile(nnodes_list, 0.2)
                        high = np.quantile(nnodes_list, 0.8)
                        correct_low = pred.eq(data.y.view(-1))[nnodes_list<=low].sum().item()
                        correct_medium = pred.eq(data.y.view(-1))[(nnodes_list>low)&(nnodes_list<high)].sum().item()
                        correct_high = pred.eq(data.y.view(-1))[nnodes_list>=high].sum().item()
                        print(100*correct_low/(nnodes_list<=low).sum(),
                            100*correct_medium/((nnodes_list>low) & (nnodes_list<high)).sum(),
                            100*correct_high/(nnodes_list>=high).sum())
                return 100*correct / len(loader.dataset)

        res = []
        best_val_acc = 0

        for it in range(epochs):
            if it == epochs//2:
                optimizer = torch.optim.Adam(model_syn.parameters(), lr=0.1*lr)

            model_syn.train()
            loss_all = 0

            optimizer.zero_grad()
            output = model_syn(feat_syn, adj_syn)
            
            if args.dataset in ['ogbg-molhiv','ogbg-molbbbp', 'ogbg-molbace']:
                loss = cls_criterion(output, labels_syn.view(-1, 1).float())
            else:
                loss = F.nll_loss(output, labels_syn.view(-1))
            loss.backward()
            loss_all += labels_syn.size(0) * loss.item()
            optimizer.step()

            if use_val:
                model_real.load_state_dict(model_syn.state_dict())
                acc_val = test(self.data[2])
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    if verbose:
                        acc_train = test(self.data[1])
                        acc_test = test(self.data[3], report_metric=False)
                        print('acc_train:', acc_train, 'acc_val:', acc_val, 'acc_test:', acc_test)
                    if save:
                        torch.save(model_real.state_dict(), f'saved/{args.dataset}_{args.seed}.pt')
                    weights = deepcopy(model_real.state_dict())
            # print(f"epoch {it} loss:{loss} val_acc:{acc_val}")

        if use_val:
            model_real.load_state_dict(weights)
        else:
            best_val_acc = test(self.data[2])
        acc_train = test(self.data[1])
        acc_test = test(self.data[3], report_metric=False)
        # print([acc_train, best_val_acc, acc_test])
        return [acc_train, best_val_acc, acc_test]


    def train(self):
        dataset = self.data[0]
        args = self.args
        
        for c_i in range(dataset.num_classes):
            eigenvals_c = []
            for ipc_e in range(args.ipc):
                eigenvals = 0.
                ind_s = self.train_list[c_i][ipc_e]
                ind_e = self.train_list[c_i][ipc_e+1]
                idxs = self.real_indices_class[c_i][ind_s:ind_e]
                for _, item in enumerate(idxs):
                    graph = self.data[4][item]
                    e = graph.e
                    eigenvals += e
                eigenvals = eigenvals / len(idxs)
                eigenvals_c.append(eigenvals.cuda())
                            
            eigenvals_c = torch.cat(eigenvals_c, dim=0)
            self.eigenvals_list.append(eigenvals_c)
            
        e_syn = torch.cat(self.eigenvals_list, dim=0)
        self.eigenvals_syn = e_syn
                
        for it in range(args.epochs):
            feat_syn = self.feat_syn
            eigenvec_syn = self.eigenvec_syn
            loss = 0
            embed_list_real = []
            embed_list_syn = []
            
            # if args.dataset not in ['ogbg-molbace', 'CIFAR10']:
            if args.dataset not in ['CIFAR10']:                
                loss_eigen = 0.
                
                for c in range(dataset.num_classes):
                    sample_ipc = np.random.permutation(range(args.ipc))[0]
                    x_real, e_real, u_real, batch_ = self.get_graphs_min(sample_ipc, c, min_node_size=args.K, batch_size=args.bs_cond, to_dense=True)
                    ind = self.syn_class_indices[c]
                    x_syn = feat_syn[ind[0]: ind[1]][sample_ipc].unsqueeze(0) #1,N,d
                    u_syn = eigenvec_syn[ind[0]: ind[1]][sample_ipc].unsqueeze(0) ##1,N,K
                    
                    # eigen loss
                    co_x_trans_real = self.get_covariance_matrix(eigenvecs=u_real, x=x_real) # bkdd
                    co_x_trans_syn = self.get_covariance_matrix(eigenvecs=u_syn, x=x_syn) # 1,kdd
                    # loss_eigen_c = 0.

                    loss_tmp = F.mse_loss(input=co_x_trans_syn, target=co_x_trans_real, reduction='none') # 1kdd, bkdd
                    loss_tmp = loss_tmp.mean()
                    loss_eigen_c = loss_tmp
                    
                    loss_eigen += loss_eigen_c
                    
                    # class loss 
                    embed_sum_real = self.get_embed_sum(eigenvals=e_real, eigenvecs=u_real, x=x_real) # b,1,K b,N,d 
                    e_syn = (self.eigenvals_list[c][sample_ipc]).unsqueeze(0)
                    embed_sum_syn = self.get_embed_sum(eigenvals=e_syn, eigenvecs=u_syn, x=x_syn) # 1,N,d
                    if args.pooling == 'sum':
                        embed_sum_syn = embed_sum_syn.sum(1)
                        embed_sum_real = embed_sum_real.sum(1)
                    embed_mean_syn_c = embed_sum_syn.mean(0)
                    embed_mean_real_c = embed_sum_real.mean(0)
                    embed_list_syn.append(embed_mean_syn_c)
                    embed_list_real.append(embed_mean_real_c)
                
                # eigen_loss
                loss += args.alpha * loss_eigen
                                
                # class_loss
                embed_mean_real = torch.stack(embed_list_real, dim=0) # c,d
                embed_mean_syn = torch.stack(embed_list_syn, dim=0) # c,d
                embed_mean_real = F.normalize(input=embed_mean_real, p=2, dim=1)
                embed_mean_syn = F.normalize(input=embed_mean_syn, p=2, dim=1)
                
                cov_embed = embed_mean_real @ embed_mean_syn.T
                iden = torch.eye(dataset.num_classes).cuda()
                class_loss = F.mse_loss(cov_embed, iden)
                loss += args.beta * class_loss

                # norm_loss
                orthog_syn = torch.bmm(u_syn.permute(0,2,1), u_syn) # # m,k,n @ m,n,k = m,k,k
                iden = torch.eye(args.K).cuda() # k,k
                orthog_norm = F.mse_loss(orthog_syn, iden.unsqueeze(0), reduction='none') # m,k,k
                orthog_norm = orthog_norm.mean(dim=[1,2]).sum()
                loss += args.gamma * orthog_norm
            
            # print(f"Epoch:{it}, loss:{loss}")
            if (it == 0) or (it == (args.epochs - 1)):
                print(f"epoch: {it}")
                print(f"eigen_match_loss: {loss_eigen}")
                print(f"args.alpha * eigen_match_loss: {args.alpha * loss_eigen}")
                print(f"class_loss: {class_loss}")
                print(f"args.beta * class_loss: {args.beta * class_loss}")
                print(f"orthog_norm: {orthog_norm}")
                print(f"args.gamma * orthog_norm: {args.gamma * orthog_norm}")
            
            self.optimizer_feat.zero_grad()
            self.optimizer_eigenvec.zero_grad()
            loss.backward()
            # update U:
            if it % (args.e1 + args.e2) < args.e1:
                self.optimizer_eigenvec.step()
            else:
                self.optimizer_feat.step()

            if it == 400:
                self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=0.1*args.lr_feat)
                self.optimizer_eigenvec = torch.optim.Adam([self.eigenvec_syn], lr=0.1*args.lr_eigenvec)

        res = []
        runs = 3
        
        for _ in range(runs):
            res.append(self.test(epochs=args.model_epoch))
        res = np.array(res)
        print('Mean Train/Val/TestAcc:', res.mean(0))
        print('Std Train/Val/TestAcc:', res.std(0))
        
        if args.save:
            feat_syn = self.feat_syn.detach() # b,N',d
            eigenvecs_syn = self.eigenvec_syn.detach() # b,N',K
            eigenvals_syn = self.eigenvals_syn # b,K
            adj_syn = eigenvecs_syn @ torch.diag_embed(eigenvals_syn) @ eigenvecs_syn.permute(0,2,1) # b,N'N'

            torch.save(
                feat_syn,
                f"saved/feat_{args.dataset}_{args.ipc}.pt",
            )
            torch.save(
                adj_syn, f"saved/adj_{args.dataset}_{args.ipc}.pt"
            )

        return res.mean(0)[2]


    def get_embed_sum(self, eigenvals, eigenvecs, x):
        x_trans = eigenvecs.permute(0,2,1) @ x  # b,K,N @ b,N,d = b,K,d        
        eigenvals_diag = torch.diag_embed(eigenvals.squeeze(1)) # b,1,K -> b,k -> b,k,k
        result = torch.bmm(eigenvals_diag, x_trans) # b,K,K @ b,K,d
        
        embed_sum = torch.bmm(eigenvecs, result) # b,n,k @ b,k,d = b,n,d
        return embed_sum


    def get_covariance_matrix(self, eigenvecs, x):
        x_trans = torch.bmm(eigenvecs.permute(0,2,1), x)  # b,K,N @ b,N,d = b,K,d
        x_trans = F.normalize(input=x_trans, p=2, dim=2) # b,K,d
        x_trans_unsqueeze = x_trans.unsqueeze(2)  # b,K,1,d
        co_matrix = x_trans_unsqueeze.permute(0,1,3,2) @ x_trans_unsqueeze  # b,K,d,1 @ b,K,1,d = b,k,d,d
        return co_matrix


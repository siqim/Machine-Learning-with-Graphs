# -*- coding: utf-8 -*-

"""
Created on December 30, 2020

@author: Siqi Miao
"""

import torch
from torch_sparse import SparseTensor
import torch_geometric.transforms as T

from pathlib2 import Path
import scipy.io as sio
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class Dataset(object):
    def __init__(self, root, name, make_edge_index=False):

        self.root = root
        self.name = name
        self.make_edge_index = make_edge_index

        self.num_classes = None
        self.split_idx = None
        self.x = None
        self.y = None
        self.adj_t = None
        self.edge_index = None
        self.num_nodes = None
        self.criterion = None
        self.metric = None

        self.heterophily_dataset = ['chameleon', 'actor']

        if name == 'ogb':
            self.setup_ogb()
        elif name == 'wiki':
            self.setup_wiki()
        elif name in self.heterophily_dataset:
            self.setup_geom()
        else:
            raise KeyboardInterrupt

    def setup_ogb(self):

        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.root, transform=T.ToSparseTensor())
        data = dataset[0]

        self.metric = 'Accuracy'
        self.num_classes = dataset.num_classes
        self.split_idx = dataset.get_idx_split()

        self.x = data.x
        self.y = data.y
        self.adj_t = data.adj_t.to_symmetric()
        self.num_nodes = data.num_nodes

        if self.make_edge_index:
            row = self.adj_t.storage.row()
            col = self.adj_t.storage.col()
            self.edge_index = torch.stack((row, col), dim=0)

        self.criterion = torch.nn.CrossEntropyLoss()

    def setup_wiki(self):

        mat = sio.loadmat(self.root / 'wiki' / 'POS.mat')

        self.metric = 'MicroF1'
        self.num_nodes = 4777
        self.num_classes = 40

        adj_t = mat['network'].tocoo()
        self.adj_t = SparseTensor(row=torch.LongTensor(adj_t.row), col=torch.LongTensor(adj_t.col),
                                  sparse_sizes=(self.num_nodes, self.num_nodes))

        if self.make_edge_index:
            row = self.adj_t.storage.row()
            col = self.adj_t.storage.col()
            self.edge_index = torch.stack((row, col), dim=0)

        self.y = torch.from_numpy(mat['group'].todense()).float()
        idx = torch.arange(self.y.shape[0]).view(-1, 1)
        train_idx, _, test_idx, _ = iterative_train_test_split(idx, self.y, test_size=0.1)
        self.split_idx = {'train': train_idx.view(-1), 'valid': test_idx.view(-1), 'test': test_idx.view(-1)}

        self.criterion = torch.nn.BCEWithLogitsLoss()  # for multi-label classification

    def setup_geom(self):
        edge_file = self.root / self.name / 'out1_graph_edges.txt'
        feature_label_file = self.root / self.name / 'out1_node_feature_label.txt'

        self.metric = 'Accuracy'

        edges = edge_file.open('r').readlines()[1:]
        edges = torch.LongTensor([(lambda x: [int(x[0]), int(x[1])])(edge.strip().split('\t')) for edge in edges])
        self.num_nodes = torch.max(edges).item() + 1
        self.adj_t = SparseTensor(row=torch.LongTensor(edges[:, 0]), col=torch.LongTensor(edges[:, 1]),
                                  sparse_sizes=(self.num_nodes, self.num_nodes))
        # self.adj_t = self.adj_t.to_symmetric()

        if self.make_edge_index:
            self.edge_index = edges.t()

        idx = []
        x = []
        y = []
        xy = feature_label_file.open('r').readlines()[1:]
        for line in xy:
            node_id, feature, label = line.strip().split('\t')
            idx.append(int(node_id))

            if self.name == 'actor':
                one_hot = torch.zeros(932)
                pos_with_ones = list(map(int, feature.split(',')))
                one_hot[pos_with_ones] = 1
                x.append(one_hot.int().tolist())
            else:
                x.append(list(map(int, feature.split(','))))
            y.append(int(label))

        _, indices = torch.sort(torch.LongTensor(idx))
        self.x = torch.LongTensor(x)[indices]
        self.y = torch.LongTensor(y).view(-1, 1)[indices]
        self.num_classes = torch.max(self.y).item() + 1

        idx = torch.arange(self.y.shape[0]).view(-1, 1)
        train_idx, val_test_idx = train_test_split(idx, test_size=0.4, stratify=self.y)
        val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, stratify=self.y[val_test_idx.squeeze()])
        self.split_idx = {'train': train_idx.view(-1), 'valid': val_idx.view(-1), 'test': test_idx.view(-1)}

        self.criterion = torch.nn.CrossEntropyLoss()

    def eval(self, y_true, logits, split_idx):

        if self.name == 'ogb':
            evaluator = Evaluator(name='ogbn-arxiv')
            y_pred = logits.argmax(dim=1, keepdim=True)
            train_acc = evaluator.eval({
                'y_true': y_true[split_idx['train']],
                'y_pred': y_pred[split_idx['train']],
            })['acc']
            valid_acc = evaluator.eval({
                'y_true': y_true[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']],
            })['acc']
            test_acc = evaluator.eval({
                'y_true': y_true[split_idx['test']],
                'y_pred': y_pred[split_idx['test']],
            })['acc']
            return train_acc, valid_acc, test_acc

        elif self.name == 'wiki':
            y_pred = torch.sigmoid(logits) > 0.5
            train_f1 = f1_score(y_true[split_idx['train']], y_pred[split_idx['train']], average='micro')
            valid_f1 = f1_score(y_true[split_idx['valid']], y_pred[split_idx['valid']], average='micro')
            test_f1 = f1_score(y_true[split_idx['test']], y_pred[split_idx['test']], average='micro')
            return train_f1, valid_f1, test_f1

        elif self.name in self.heterophily_dataset:
            y_pred = logits.argmax(dim=1, keepdim=True)
            train_acc = accuracy_score(y_true[split_idx['train']], y_pred[split_idx['train']])
            valid_acc = accuracy_score(y_true[split_idx['valid']], y_pred[split_idx['valid']])
            test_acc = accuracy_score(y_true[split_idx['test']], y_pred[split_idx['test']])
            return train_acc, valid_acc, test_acc


if __name__ == '__main__':
    data = Dataset(root=Path('../dataset'), name='ogb', make_edge_index=True)

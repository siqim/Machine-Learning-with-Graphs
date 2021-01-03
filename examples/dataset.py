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
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class Dataset(object):
    def __init__(self, root, name):

        self.root = root
        self.num_classes = None
        self.split_idx = None
        self.x = None
        self.y = None
        self.adj_t = None
        self.edge_index = None
        self.num_nodes = None
        self.criterion = None
        self.metric = None
        self.name = name

        if name == 'ogb':
            self.setup_ogb()
        elif name == 'wiki':
            self.setup_wiki()
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
        self.edge_index = data.edge_index
        self.num_nodes = data.num_nodes

        self.criterion = torch.nn.CrossEntropyLoss()

    def setup_wiki(self):

        mat = sio.loadmat(self.root / 'wiki' / 'POS.mat')

        self.metric = 'MicroF1'
        self.num_nodes = 4777
        self.num_classes = 40

        adj_t = mat['network'].tocoo()
        self.adj_t = SparseTensor(row=torch.LongTensor(adj_t.row), col=torch.LongTensor(adj_t.col),
                                  sparse_sizes=(self.num_nodes, self.num_nodes))

        self.y = torch.from_numpy(mat['group'].todense()).float()
        X = torch.arange(self.y.shape[0]).view(-1, 1)
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, self.y, test_size=0.1)
        self.split_idx = {'train': X_train.view(-1), 'valid': X_test.view(-1), 'test': X_test.view(-1)}

        self.criterion = torch.nn.BCEWithLogitsLoss()  # for multi-label classification

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


if __name__ == '__main__':
    data = Dataset(root=Path('../dataset'), name='wiki')

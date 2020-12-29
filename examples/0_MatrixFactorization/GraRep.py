# -*- coding: utf-8 -*-

"""
Created on December 29, 2020

@author: Siqi Miao
"""

from scipy.sparse.linalg import svds

import torch
import torch.nn as nn

import torch_sparse
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class GraRep(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, k):
        super().__init__()
        self.H = nn.Embedding(num_embeddings, embedding_dim)
        self.C = nn.Embedding(num_embeddings, embedding_dim)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def construct_Ak(self, adj, k):
        assert k >= 1
        deg_inv = 1 / torch_sparse.sum(adj, dim=1).view(-1, 1)
        A_tilde = torch_sparse.mul(adj, deg_inv)

        A = {1: A_tilde}
        if k >= 2:
            for i in range(2, k + 1):
                A[i] = A.get(i - 1).matmul(A_tilde)
        return A

    @staticmethod
    def solve_with_SVD(self, A, num_nodes, embedding_dim, lamb=1):

        W = []
        for k, Ak in A.items():
            tau_k = Ak.sum(dim=0).view(1, -1)
            Xk = torch_sparse.mul(Ak, num_nodes / (tau_k * lamb))
            temp = torch.log(Xk.storage.value())
            temp[temp < 0] = 0
            Xk.storage._value = temp

            Xk = Xk.to_scipy('coo')
            u, s, vt = svds(Xk, k=embedding_dim)  # torch.svd_lowrank does not work due to a bug
            Wk = torch.tensor(u * s ** (-0.5))
            W.append(Wk)

        return torch.cat(W, dim=0)

    def forward(self, i, j):
        h = self.H(i)
        c = self.C(j)
        return self.sigmoid((h * c).sum(dim=1))

    def classifier(self, w):
        pass


def train(model, data, train_idx, optimizer, criterion):
    model.train()

    model.zero_grad()
    outputs = model(data.x, data.adj_t)[train_idx]
    loss = criterion(outputs, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    outputs = model(data.x, data.adj_t)
    y_pred = outputs.argmax(dim=1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

hidden_channels = 256
num_layers = 3
dropout = 0.5
epochs = 500
log_steps = 10
lr = 0.01

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='../../dataset', transform=T.ToSparseTensor())

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

model = GCN(data.num_features, dataset.num_classes, hidden_channels, num_layers, dropout).to(device)

evaluator = Evaluator(name='ogbn-arxiv')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

data.adj_t = torch_sparse.fill_diag(data.adj_t, 1)
deg = torch_sparse.sum(data.adj_t, 0).pow_(-0.5)
data.adj_t = torch_sparse.mul(data.adj_t, deg.view(-1, 1))
data.adj_t = torch_sparse.mul(data.adj_t, deg.view(1, -1))

test_scores = []
for epoch in range(1, 1 + epochs):
    loss = train(model, data, train_idx, optimizer, criterion)
    result = test(model, data, split_idx, evaluator)

    if epoch % log_steps == 0:
        train_acc, valid_acc, test_acc = result
        test_scores.append(test_acc)
        print(f'Run: {1:02d}, '
              f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')
print(f"Best test accuracy: {max(test_scores) * 100:.2f}%")

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

torch_sparse.sum(data.adj_t, 1)

import torch_sparse
from torch_sparse import SparseTensor

mat = torch.FloatTensor(
    [[0, 1, 0],
     [0, 0, 1],
     [1, 1, 0]]
)

adj = SparseTensor.from_dense(mat)

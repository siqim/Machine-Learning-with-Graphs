# -*- coding: utf-8 -*-

"""
Created on 2021/3/12

@author: Siqi Miao
"""

import sys
sys.path.append("../")

import math
from pathlib2 import Path
from dataset import Dataset

import torch
import torch.nn as nn
import torch_sparse


class GCNConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias)

    def forward(self, x, adj_t):
        x = x.matmul(self.weight)
        return torch_sparse.matmul(adj_t, x) + self.bias


class GCN(nn.Module):

    def __init__(self, num_layers, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x, adj_t):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        out = self.convs[-1](x, adj_t)
        return out


def train(model, data, optimizer):
    model.train()
    train_idx = data.split_idx['train']

    model.zero_grad()
    outputs = model(data.x, data.adj_t)[train_idx]

    loss = data.criterion(outputs, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits = model(data.x, data.adj_t)
    return data.eval(data.y.cpu(), logits.cpu(), data.split_idx)


def gcn_norm(adj_t):
    adj_t = torch_sparse.fill_diag(adj_t, 1)  # add self-loop

    deg = torch_sparse.sum(adj_t, dim=1).pow_(-0.5)  # compute normalized degree matrix
    deg.masked_fill_(deg == float('inf'), 0.)  # for numerical stability

    adj_t = torch_sparse.mul(adj_t, deg.view(-1, 1))  # row-wise mul
    adj_t = torch_sparse.mul(adj_t, deg.view(1, -1))  # col-wise mul
    return adj_t


def main():
    dataset = 'ogb'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    num_layers = 3
    hidden_channels = 256
    epochs = 50
    lr = 0.01
    log_steps = 1

    data = Dataset(root=Path('../../dataset'), name=dataset)
    data.adj_t = gcn_norm(data.adj_t).to(device)
    data.x = data.x.to(device)
    model = GCN(num_layers, data.x.shape[1], hidden_channels, data.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    test_scores = []
    for epoch in range(epochs + 1):
        loss = train(model, data, optimizer)
        result = test(model, data)

        if epoch % log_steps == 0:
            train_res, valid_res, test_res = result
            test_scores.append(test_res)
            print(f'Run: {1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Metric: {data.metric}',
                  f'Train: {100 * train_res:.2f}%, '
                  f'Valid: {100 * valid_res:.2f}% '
                  f'Test: {100 * test_res:.2f}%')
    print(f"Best test accuracy: {max(test_scores) * 100:.2f}%")


if __name__ == '__main__':
    main()

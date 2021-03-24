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


class GATConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_heads):
        super(GATConv, self).__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.linear = nn.Linear(in_channels, out_channels * num_heads)
        self.att_l = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, num_heads, out_channels))

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, adj_t):
        x = self.linear(x).view(-1, self.num_heads, self.out_channels)  # N x H x C

        alpha_l = (self.att_l * x).sum(dim=-1)  # N x H
        alpha_r = (self.att_r * x).sum(dim=-1)  # N x H

        # E x N         E x H x C
        alpha_lifted, x_lifted = self.lift(alpha_l, alpha_r, x, adj_t)
        alpha_per_edge = self.leakyrelu(alpha_lifted)  # E x N




    def lift(self, alpha_l, alpha_r, x, adj_t):
        src_nodes_index = adj_t.storage.row()
        trg_nodes_index = adj_t.storage.col()

        alpha_lifted = alpha_l.index_select(0, src_nodes_index) + alpha_r.index_select(0, trg_nodes_index)
        x_lifted = x.index_select(0, src_nodes_index)
        return alpha_lifted, x_lifted




class GAT(nn.Module):

    def __init__(self, num_layers, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, 2))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, 2))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GATConv(hidden_channels, out_channels, 2))

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


def main():
    dataset = 'ogb'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_layers = 3
    hidden_channels = 32
    epochs = 50
    lr = 0.01
    log_steps = 1

    data = Dataset(root=Path('../../dataset'), name=dataset)
    data.x = data.x.to(device)
    model = GAT(num_layers, data.x.shape[1], hidden_channels, data.num_classes).to(device)
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

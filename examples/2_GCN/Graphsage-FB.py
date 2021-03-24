# -*- coding: utf-8 -*-

"""
Created on 2021/3/12

@author: Siqi Miao
"""

import sys
sys.path.append("../")

from pathlib2 import Path
from dataset import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse


class SageConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SageConv, self).__init__()

        self.linear_1 = nn.Linear(in_channels, out_channels, bias=False)
        self.linear_2 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, h, adj_t):
        h_n = torch_sparse.matmul(adj_t, h, reduce='mean')
        # h = self.linear(torch.cat((h, h_n), dim=1))  # to make my life easier, I don't concat here.
        h = self.linear_1(h) + self.linear_2(h_n)
        h = F.normalize(h, p=2, dim=1)
        return h


class Graphsage(nn.Module):

    def __init__(self, num_layers, in_channels, hidden_channels, out_channels):
        super(Graphsage, self).__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(SageConv(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SageConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(SageConv(hidden_channels, out_channels))

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    num_layers = 3
    hidden_channels = 256
    epochs = 200
    lr = 0.01
    log_steps = 1

    data = Dataset(root=Path('../../dataset'), name=dataset)
    data.x = data.x.to(device)
    model = Graphsage(num_layers, data.x.shape[1], hidden_channels, data.num_classes).to(device)
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

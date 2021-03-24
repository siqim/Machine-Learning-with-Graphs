# -*- coding: utf-8 -*-

"""
Created on 2021/3/12

@author: Siqi Miao
"""

import sys
sys.path.append("../")

from tqdm import tqdm
from pathlib2 import Path
from dataset import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_sparse
from torch_geometric.data import NeighborSampler


class SageConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SageConv, self).__init__()

        self.linear_1 = nn.Linear(in_channels, out_channels, bias=False)
        self.linear_2 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, h, h_target, adj_t):
        adj_t = adj_t.set_value(None, layout=None)  # torch_sparse.matmul will throw an error without this line
        h_n = torch_sparse.matmul(adj_t, h, reduce='mean')
        # h = self.linear(torch.cat((h_target, h_n), dim=1))  # to make my life easier, I don't concat here.
        h = self.linear_1(h_target) + self.linear_2(h_n)
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

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # get features of target nodes for aggregation
            x = self.convs[i](x, x_target, edge_index)
            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = self.relu(x)
                x = self.dropout(x)
        return x


def train(model, data, data_loader, optimizer, device):
    model.train()

    total_loss = total_iters = 0
    for batch_size, n_id, adjs in tqdm(data_loader):
        # n_id includes all relevant nodes in the computational graph,
        # and target nodes are included at the beginning of it.
        adjs = [adj.to(device) for adj in adjs]
        model.zero_grad()
        outputs = model(data.x[n_id], adjs)

        loss = data.criterion(outputs, data.y.squeeze(1)[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iters += 1

    return total_loss / total_iters


@torch.no_grad()
def test(model, data, data_loader, device):
    model.eval()

    all_logits = []
    for batch_size, n_id, adjs in tqdm(data_loader):
        adjs = [adj.to(device) for adj in adjs]
        logits = model(data.x[n_id], adjs)
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)
    return data.eval(data.y.cpu(), all_logits.cpu(), data.split_idx)


def main():
    dataset = 'ogb'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    num_layers = 3
    sizes = [10 for _ in range(num_layers)]
    hidden_channels = 256
    epochs = 200
    lr = 0.01
    log_steps = 1
    batch_size = 1024*4

    data = Dataset(root=Path('../../dataset'), name=dataset)
    train_loader = NeighborSampler(data.adj_t, sizes=sizes, node_idx=data.split_idx['train'],
                                   batch_size=batch_size, shuffle=True)
    subgraph_loader = NeighborSampler(data.adj_t, sizes=[-1, -1, -1], node_idx=None,
                                      batch_size=batch_size, shuffle=False)

    model = Graphsage(num_layers, data.x.shape[1], hidden_channels, data.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    test_scores = []
    for epoch in range(epochs + 1):
        loss = train(model, data, train_loader, optimizer, device)
        result = test(model, data, subgraph_loader, device)

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

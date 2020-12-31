# -*- coding: utf-8 -*-

"""
Created on December 29, 2020

@author: Siqi Miao
"""

import sys
sys.path.append("../")

from pathlib2 import Path
from dataset import Dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_geometric.utils import negative_sampling


class GF(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, x_dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + x_dim, embedding_dim + x_dim),
            nn.BatchNorm1d(embedding_dim + x_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(embedding_dim + x_dim, num_classes)
        )

    def forward_emb(self, i, j):
        h_i = self.emb(i)
        h_j = self.emb(j)
        return self.sigmoid((h_i * h_j).sum(dim=1))

    def forward(self, h):
        return self.fc(h)


def train_emb(model, edge_index, data, optimizer, batch_size, device):
    model.train()
    data_loader = DataLoader(range(edge_index.shape[1]), batch_size, shuffle=True)

    total_loss = total_examples = 0
    for perm in data_loader:
        model.zero_grad()

        pos_edge = edge_index[:, perm].to(device)  # 2 x batch_size
        pos_out = model.forward_emb(*pos_edge)
        # add 1e-15 to avoid exploding gradients
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # negative sampling on the graph
        neg_edge = negative_sampling(edge_index, num_nodes=data.num_nodes,
                                     num_neg_samples=perm.size(0), method='sparse').to(device)  # 2 x batch_size
        neg_out = model.forward_emb(*neg_edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def train(model, H, data, optimizer):
    model.train()
    train_idx = data.split_idx['train']

    model.zero_grad()
    outputs = model(H)[train_idx]

    loss = data.criterion(outputs, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, H, data):
    model.eval()
    logits = model(H)

    return data.eval(data.y.cpu(), logits.cpu(), data.split_idx)


def main():
    dataset = 'wiki'  # 'wiki' or 'ogb'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    embedding_dim = 256
    lr = 0.1  # 0.1 for wiki and 0.01 for ogb
    epochs = 200
    log_steps = 1
    batch_size = 1024 * 1024

    data = Dataset(root=Path('../../dataset'), name=dataset)
    row, col, value = data.adj_t.coo()
    edge_index = torch.stack((row, col), dim=0)

    model = GF(data.num_nodes, embedding_dim, data.num_classes,
               0 if data.x is None else data.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 1. train the embedding with negative sampling
    for iters in range(100):
        loss = train_emb(model, edge_index, data, optimizer, batch_size, device)
        if iters % log_steps == 0:
            print("Iters: {iters}, Loss: {loss:.4f}".format(iters=iters, loss=loss))

    # 2. train a classifier for the node classification task
    H = model.emb.weight
    H.requires_grad = False
    if data.x is not None:
        H = torch.cat([data.x.to(device), H], dim=1)  # if there are node features, add them
    data.y = data.y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    test_scores = []
    for epoch in range(1, 1 + epochs):
        loss = train(model, H, data, optimizer)
        result = test(model, H, data)

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

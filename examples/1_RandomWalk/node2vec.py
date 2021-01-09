# -*- coding: utf-8 -*-

"""
Created on January 04, 2021

@author: Siqi Miao
"""

import sys

sys.path.append("../")

from tqdm import tqdm
from pathlib2 import Path
from dataset import Dataset
from random_walk_numba import random_walk

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Node2Vec(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_classes, x_dim,
                 rowptr, col, p, q, walks_per_node, walk_length, context_size):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.rowptr = rowptr
        self.col = col
        self.p = p
        self.q = q
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.context_size = context_size

        self.emb = nn.Embedding(num_nodes, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + x_dim, embedding_dim + x_dim),
            nn.BatchNorm1d(embedding_dim + x_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(embedding_dim + x_dim, num_classes)
        )

    def forward_emb(self, i):
        return self.emb(i)

    def forward_clf(self, h):
        return self.fc(h)

    def get_samples(self):
        pos_rw = torch.from_numpy(random_walk(self.rowptr, self.col, self.p, self.q, self.num_nodes,
                                              self.walks_per_node, self.walk_length))
        pos_samples = self.walks2samples(pos_rw)

        neg_nodes = torch.arange(self.num_nodes)
        neg_nodes = neg_nodes.repeat(self.walks_per_node)  # pos_samples:neg_samples = 1:1
        neg_rw = torch.randint(self.num_nodes, (neg_nodes.size(0), self.walk_length))
        neg_rw = torch.cat([neg_nodes.view(-1, 1), neg_rw], dim=-1)
        neg_samples = self.walks2samples(neg_rw)
        return pos_samples, neg_samples

    def walks2samples(self, rw):
        walks = []
        num_walks_per_rw = self.walk_length - self.context_size
        for j in range(num_walks_per_rw + 1):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def loss(self, pos_samples, neg_samples):
        """
        Modified from torch_geometric.nn.Node2Vec.loss,
        which computes the skip-gram loss.
        """

        # Positive loss.
        start, rest = pos_samples[:, [0]], pos_samples[:, 1:]
        h_start = self.emb(start)
        h_rest = self.emb(rest)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + 1e-15).mean()

        # Negative loss.
        start, rest = neg_samples[:, [0]], neg_samples[:, 1:]

        h_start = self.emb(start)
        h_rest = self.emb(rest)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(torch.sigmoid(-out) + 1e-15).mean()

        return pos_loss + neg_loss


def train_emb(model, data_loader, pos_samples, neg_samples, optimizer, device):
    model.train()

    total_loss = 0
    pbar = tqdm(data_loader)
    for perm in pbar:
        model.zero_grad()

        pos_batch = pos_samples[perm].to(device)
        neg_batch = neg_samples[perm].to(device)

        loss = model.loss(pos_batch, neg_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description('Loss: {loss:.4f}'.format(loss=loss.item()))

    return total_loss / len(data_loader)


def train_clf(model, H, data, optimizer):
    model.train()
    train_idx = data.split_idx['train']

    model.zero_grad()
    outputs = model.forward_clf(H)[train_idx]

    loss = data.criterion(outputs, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_clf(model, H, data):
    model.eval()
    logits = model.forward_clf(H)
    return data.eval(data.y.cpu(), logits.cpu(), data.split_idx)


def main():
    dataset = 'ogb'  # 'wiki' or 'ogb'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    p = 1
    q = 1
    walk_length = 20
    context_size = 10
    walks_per_node = 40
    skip_gram_epochs = 5

    embedding_dim = 256
    lr = 0.01
    clf_epochs = 200
    log_steps = 1
    batch_size = 128 * 1024

    data = Dataset(root=Path('../../dataset'), name=dataset)
    rowptr, col, _ = data.adj_t.csr()  # use rowptr to retrieve neighbors in O(1)
    rowptr = rowptr.numpy()
    col = col.numpy()

    model = Node2Vec(data.num_nodes, embedding_dim, data.num_classes,
                     0 if data.x is None else data.x.shape[1],
                     rowptr, col, p, q, walks_per_node, walk_length, context_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print('1. Generating samples for the skip-gram model...')
    pos_samples, neg_samples = model.get_samples()
    data_loader = DataLoader(range(pos_samples.shape[0]), batch_size, shuffle=True)

    print('2. Training the skip-gram model...')
    for epoch in range(1, 1 + skip_gram_epochs):
        loss = train_emb(model, data_loader, pos_samples, neg_samples, optimizer, device)
        if epoch % log_steps == 0:
            print("Epochs: {epoch}, AvgLoss: {loss:.4f}".format(epoch=epoch, loss=loss))

    print('3. Training the classifier...')
    H = model.emb.weight.detach()
    H = torch.cat([data.x.to(device), H], dim=1) if data.x is not None else H  # if there are node features, add them

    data.y = data.y.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    test_scores = []
    for epoch in range(1, 1 + clf_epochs):
        loss = train_clf(model, H, data, optimizer)
        result = test_clf(model, H, data)

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

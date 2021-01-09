# -*- coding: utf-8 -*-

"""
Created on January 02, 2021

@author: Siqi Miao
"""

import sys

sys.path.append("../")

from pathlib2 import Path
from scipy import sparse
from scipy.sparse.linalg import svds, inv
from dataset import Dataset

import torch
import torch.nn as nn


class HOPE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, beta, x_dim):
        super().__init__()

        self.beta = beta
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + x_dim, embedding_dim + x_dim),
            nn.BatchNorm1d(embedding_dim + x_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(embedding_dim + x_dim, num_classes)
        )

    def solve_with_SVD_v1(self, adj):
        # original version of HOPE, aiming to directly reconstruct S
        adj = adj.to_scipy('csc')

        # Common Neighbors A^2
        S = adj.multiply(adj)

        # Katz index
        # I = sparse.eye(self.num_embeddings, format='csc')
        # modified Katz index proposed in the paper, it doesn't make sense and doesn't work
        # S = inv(I - self.beta * adj).multiply(self.beta * adj)
        # original definition of Katz index, too expensive to compute
        # S = inv(I - self.beta * adj) - I

        u, s, vt = svds(S, k=self.embedding_dim // 2)
        Us = torch.tensor(u * s ** 0.5)
        Ut = torch.tensor(vt.T * s ** 0.5)

        return torch.cat((Us, Ut), dim=1).float()

    def solve_with_SVD_v2(self, adj):
        # modified version of HOPE, which uses a GraRep-like objective
        adj = adj.to_scipy('csc')
        adj_2 = adj.multiply(adj)

        adj[adj > 0] = 1e3  # simplify the problem
        u, s, vt = svds(adj, k=self.embedding_dim // 2)
        w1 = torch.tensor(u * s ** 0.5)

        adj_2[adj_2 > 0] = 1e3  # simplify the problem
        u, s, vt = svds(adj_2, k=self.embedding_dim // 2)
        w2 = torch.tensor(u * s ** 0.5)

        return torch.cat((w1, w2), dim=1).float()

    def forward(self, h):
        return self.fc(h)


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
def test(model, W, data):
    model.eval()
    logits = model(W)

    return data.eval(data.y.cpu(), logits.cpu(), data.split_idx)


def main():
    dataset = 'wiki'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    beta = 0.01  # for computing Katz index
    embedding_dim = 256
    epochs = 500
    log_steps = 5
    lr = 0.01

    data = Dataset(root=Path('../../dataset'), name=dataset)

    model = HOPE(data.num_nodes, embedding_dim, data.num_classes, beta,
                 0 if data.x is None else data.x.shape[1]).to(device)

    # 1. construct S and obtain H via truncated SVD
    H = model.solve_with_SVD_v2(data.adj_t).to(device)

    # 2. use the learned representations to train a classifier for the node classification task
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

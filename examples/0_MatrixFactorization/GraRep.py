# -*- coding: utf-8 -*-

"""
Created on December 29, 2020

@author: Siqi Miao
"""

import sys

sys.path.append("../")

from pathlib2 import Path
from scipy.sparse.linalg import svds
from dataset import Dataset

import torch
import torch.nn as nn
import torch_sparse


class GraRep(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, k, x_dim):
        super().__init__()

        self.k = k
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        assert self.k >= 1

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * k + x_dim, embedding_dim * k + x_dim),
            nn.BatchNorm1d(embedding_dim * k + x_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(embedding_dim * k + x_dim, num_classes)
        )

    def solve_with_SVD_v1(self, adj, lamb=1):
        # the original NCE version of GraRep

        # first construct A^k with normalization
        deg_inv = 1 / (torch_sparse.sum(adj, dim=1).view(-1, 1))
        A_tilde = torch_sparse.mul(adj, deg_inv)
        A = [A_tilde]
        if self.k >= 2:
            for i in range(1, self.k):
                A.append(A[i - 1].matmul(A_tilde))

        # then obtain W via low-rank SVD
        W = []
        C = []
        for k, Ak in enumerate(A, 1):
            print('solving SVD with k={k}'.format(k=k))
            tau_k = Ak.sum(dim=0).view(1, -1)

            Xk = torch_sparse.mul(Ak, self.num_embeddings / (tau_k * lamb))
            temp = torch.log(Xk.storage.value() + 1e-15)
            temp[temp < 0] = 0
            Xk.storage._value = temp

            Xk = Xk.to_scipy('coo')
            u, s, vt = svds(Xk, k=self.embedding_dim)  # torch.svd_lowrank does not work due to a bug
            Wk = torch.tensor(u * s ** 0.5)
            Ck = torch.tensor(vt.T * s ** 0.5)
            W.append(Wk)
            C.append(Ck)

        W = torch.cat(W, dim=1)
        C = torch.cat(C, dim=1)
        H = torch.cat((W, C), dim=1)
        return W

    def solve_with_SVD_v2(self, adj, lamb=1):
        # NS version of GraRep

        # first construct A^k with normalization
        deg_inv = 1 / (torch_sparse.sum(adj, dim=1).view(-1, 1))
        A_tilde = torch_sparse.mul(adj, deg_inv)
        A = [A_tilde]
        if self.k >= 2:
            for i in range(1, self.k):
                A.append(A[i - 1].matmul(A_tilde))

        # then obtain W via low-rank SVD
        W = []
        for k, Ak in enumerate(A, 1):
            print('solving SVD with k={k}'.format(k=k))
            tau_k = Ak.sum(dim=0).view(1, -1)

            # remove the conditional probability term in the objective function
            # to make it a pure negative sampling model
            temp = Ak.storage.value()
            temp[temp > 0] = 1
            temp[temp <= 0] = 0
            Ak.storage._value = temp

            Xk = torch_sparse.mul(Ak, self.num_embeddings / (tau_k * lamb))
            temp = torch.log(Xk.storage.value() + 1e-15)
            temp[temp < 0] = 0
            Xk.storage._value = temp

            Xk = Xk.to_scipy('coo')
            u, s, vt = svds(Xk, k=self.embedding_dim)  # torch.svd_lowrank does not work due to a bug
            Wk = torch.tensor(u * s ** 0.5)
            W.append(Wk)

        return torch.cat(W, dim=1)

    def solve_with_SVD_v3(self, adj, lamb=1):
        # the version that directly adopts the objective function from the paper:
        # Neural Word Embedding as Implicit Matrix Factorization

        # first construct A^k without normalization
        A = [adj]
        if self.k >= 2:
            for i in range(1, self.k):
                A.append(A[i - 1].matmul(adj))

        # then obtain W via low-rank SVD
        W = []
        for k, Ak in enumerate(A, 1):
            print('solving SVD with k={k}'.format(k=k))
            num_c = Ak.sum(dim=0).view(1, -1)
            num_w = Ak.sum(dim=1).view(-1, 1)
            D = Ak.sum()

            Xk = torch_sparse.mul(Ak, 1 / num_w)
            Xk = torch_sparse.mul(Xk, D / (num_c * lamb))
            temp = torch.log(Xk.storage.value() + 1e-15)
            temp[temp < 0] = 0
            Xk.storage._value = temp

            Xk = Xk.to_scipy('coo')
            u, s, vt = svds(Xk, k=self.embedding_dim)  # torch.svd_lowrank does not work due to a bug
            Wk = torch.tensor(u * s ** 0.5)
            W.append(Wk)

        return torch.cat(W, dim=1)

    def forward(self, w):
        return self.fc(w)


def train(model, W, data, optimizer):
    model.train()
    train_idx = data.split_idx['train']

    model.zero_grad()
    outputs = model(W)[train_idx]

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

    k = 2
    embedding_dim = 256 // k
    epochs = 500
    log_steps = 5
    lr = 0.01

    data = Dataset(root=Path('../../dataset'), name=dataset)

    model = GraRep(data.num_nodes, embedding_dim, data.num_classes, k,
                   0 if data.x is None else data.x.shape[1]).to(device)
    # 1. construct A^k and obtain W via truncated SVD
    W = model.solve_with_SVD_v1(data.adj_t).to(device)

    # 2. use the learned representations to train a classifier for the node classification task
    if data.x is not None:
        W = torch.cat([data.x.to(device), W], dim=1)  # if there are node features, add them
    data.y = data.y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    test_scores = []
    for epoch in range(1, 1 + epochs):
        loss = train(model, W, data, optimizer)
        result = test(model, W, data)

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

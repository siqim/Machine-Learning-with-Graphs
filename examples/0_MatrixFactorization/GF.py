# -*- coding: utf-8 -*-

"""
Created on December 29, 2020

@author: Siqi Miao
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


class GF(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, i, j):
        h_i = self.emb(i)
        h_j = self.emb(j)
        return self.sigmoid((h_i * h_j).sum(dim=1))


def train(model, optimizer, split_edge, data, batch_size, device):
    model.train()

    pos_train_edge = split_edge['train']['edge']
    data_loader = DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True)

    total_loss = total_examples = 0
    for perm in data_loader:
        model.zero_grad()

        pos_edge = pos_train_edge[perm].t().to(device)  # 2 x batch_size
        pos_out = model(*pos_edge)
        # add 1e-15 to avoid exploding gradients
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # negative sampling on the graph
        neg_edge = negative_sampling(data.edge_index, num_nodes=data.num_nodes,
                                     num_neg_samples=perm.size(0), method='dense').to(device)  # 2 x batch_size
        neg_out = model(*neg_edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, split_edge, evaluator, device):
    model.eval()

    pos_train_edge = split_edge['train']['edge'].t().to(device)
    pos_valid_edge = split_edge['valid']['edge'].t().to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].t().to(device)
    pos_test_edge = split_edge['test']['edge'].t().to(device)
    neg_test_edge = split_edge['test']['edge_neg'].t().to(device)

    pos_train_pred = model(*pos_train_edge)
    pos_valid_pred = model(*pos_valid_edge)
    neg_valid_pred = model(*neg_valid_edge)
    pos_test_pred = model(*pos_test_edge)
    neg_test_pred = model(*neg_test_edge)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi', root='../../dataset')
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    embedding_dim = 256
    lr = 0.01
    epochs = 100
    log_steps = 5
    batch_size = 64 * 1024

    model = GF(data.num_nodes, embedding_dim).to(device)
    evaluator = Evaluator(name='ogbl-ddi')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, 1 + epochs):

        loss = train(model, optimizer, split_edge, data, batch_size, device)
        results = test(model, split_edge, evaluator, device)

        if epoch % log_steps == 0:
            for key, result in results.items():
                train_hits, valid_hits, test_hits = result
                print(key)
                print(f'Run: {1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_hits:.2f}%, '
                      f'Valid: {100 * valid_hits:.2f}%, '
                      f'Test: {100 * test_hits:.2f}%')
            print('---')


if __name__ == '__main__':
    main()

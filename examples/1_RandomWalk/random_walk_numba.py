# -*- coding: utf-8 -*-

"""
Created on January 04, 2021

@author: Siqi Miao
"""

import sys

sys.path.append("../")

import time
import numpy as np
from pathlib2 import Path
from dataset import Dataset
from numba import njit, prange


@njit(nogil=True)
def walk_from_start(rowptr, col, start, walk_length, p, q):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=np.int64)
    walk[0] = start
    walk[1] = sample_neighbors_uniformly(rowptr, col, start)

    for j in range(2, walk_length):
        while True:
            new_node = sample_neighbors_uniformly(rowptr, col, walk[j - 1])
            r = np.random.rand()
            if new_node == walk[j - 2]:
                # back to the previous node
                if r < prob_0:
                    break
            elif is_neighbor(rowptr, col, walk[j - 2], new_node):
                # distance 1
                if r < prob_1:
                    break
            elif r < prob_2:
                # distance 2
                break
        walk[j] = new_node

    return walk


@njit(nogil=True)
def is_neighbor(rowptr, col, a, b):
    # O(log(d_bar))
    a_neighbs = get_neighbors(rowptr, col, a)
    if b <= a_neighbs[-1] and np.searchsorted(a_neighbs, b, side='right'):
        return True
    else:
        return False


@njit(nogil=True)
def sample_neighbors_uniformly(rowptr, col, a):
    # O(1)
    neighbs = get_neighbors(rowptr, col, a)
    return np.random.choice(neighbs)


@njit(nogil=True)
def get_neighbors(rowptr, col, a):
    # O(1)
    return col[rowptr[a]:rowptr[a + 1]]


@njit(nogil=True, parallel=True)
def random_walk(rowptr, col, p, q, num_nodes, walks_per_node, walk_length):
    walks = np.zeros((num_nodes * walks_per_node, walk_length), dtype=np.int64)
    for i in prange(num_nodes):
        for j in prange(walks_per_node):
            walks[i * walks_per_node + j] = walk_from_start(rowptr, col, i, walk_length, p, q)
    return walks


if __name__ == '__main__':
    data = Dataset(root=Path('../../dataset'), name='ogb')

    tik = time.time()
    random_walk(rowptr=data.adj_t.storage.rowptr().numpy(),
                col=data.adj_t.storage.col().numpy(),
                p=4, q=0.5, num_nodes=data.num_nodes,
                walks_per_node=10, walk_length=80)
    tok = time.time()
    print('{seconds:.2f} used'.format(seconds=tok - tik))

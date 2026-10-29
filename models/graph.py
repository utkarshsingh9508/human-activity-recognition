"""Adjacency matrices for ST-GCN with the three partitioning strategies of
Yan et al. (AAAI 2018):
  uniform  K=1  all neighbours (and the joint itself) share one weight
  distance K=2  the joint itself | its direct neighbours
  spatial  K=3  the joint itself (+ same-distance neighbours) | neighbours closer to
               the body centre (centripetal) | neighbours farther away (centrifugal)
Returns A with shape (K, V, V); column i of A[k] = how joint i averages partition k."""
import numpy as np


def shortest_paths(num_nodes, edges):
    d = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(d, 0)
    for i, j in edges:
        d[i, j] = d[j, i] = 1
    for k in range(num_nodes):                     # Floyd-Warshall, tiny graph
        d = np.minimum(d, d[:, k:k + 1] + d[k:k + 1, :])
    return d


def build_adjacency(num_nodes, edges, center, strategy="spatial"):
    hop = shortest_paths(num_nodes, edges)
    if np.isinf(hop).any():
        raise ValueError("skeleton graph is not connected")
    adj = (hop <= 1).astype(np.float32)
    norm = adj / adj.sum(axis=0, keepdims=True)
    if strategy == "uniform":
        return norm[None].astype(np.float32)
    if strategy == "distance":
        return np.stack([norm * (hop == 0), norm * (hop == 1)]).astype(np.float32)
    if strategy != "spatial":
        raise ValueError(f"unknown partition strategy {strategy!r}")
    root, close, far = (np.zeros_like(norm) for _ in range(3))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if hop[j, i] > 1:
                continue
            if hop[j, center] == hop[i, center]:
                root[j, i] = norm[j, i]
            elif hop[j, center] < hop[i, center]:
                close[j, i] = norm[j, i]
            else:
                far[j, i] = norm[j, i]
    return np.stack([root, close, far]).astype(np.float32)

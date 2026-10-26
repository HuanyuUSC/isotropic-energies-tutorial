"""Finger segmentation of the benchmark hand: geodesic distance g from the wrist; each finger is the part of
a superlevel set {g > t} that holds one fingertip alone, at the last level t before it joins another fingertip.
    labels = fingers(V, F)    # -1 = palm/wrist, 0..4 = fingers"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra


def adjacency(V, F):
    i = np.r_[F[:, 0], F[:, 1], F[:, 2]]; j = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    w = np.linalg.norm(V[i] - V[j], axis=1)
    A = sp.csr_matrix((w, (i, j)), shape=(len(V), len(V)))
    return A.maximum(A.T)


def fingers(V, F, n=5):
    """Superlevel-set persistence of g (union-find, vertices by decreasing g). The n most persistent branches
    are the fingers; each finger = the vertices its branch had collected when it merged into an older one."""
    A = adjacency(V, F).tocsr()
    wrist = np.flatnonzero(V[:, 1] < V[:, 1].min() + 0.03)
    g = dijkstra(A, indices=wrist, min_only=True)
    order = np.argsort(-g)
    parent = -np.ones(len(V), int)
    birth, members, events = {}, {}, []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for v in order:
        parent[v] = v; birth[v] = g[v]; members[v] = [v]
        for u in A.indices[A.indptr[v]:A.indptr[v + 1]]:
            if parent[u] < 0:
                continue
            a, b = find(u), find(v)
            if a == b:
                continue
            old, young = (a, b) if birth[a] >= birth[b] else (b, a)   # elder rule
            if birth[old] == g.max() and birth[young] - g[v] > 0.1 and "eldest" not in birth:
                birth["eldest"] = list(members[old])                     # the longest finger, alone
            events.append((birth[young] - g[v], young, list(members[young])))
            parent[young] = old; members[old] += members[young]; del members[young]
    events.sort(key=lambda e: -e[0])
    events = [(np.inf, order[0], birth["eldest"])] + events
    lab = np.full(len(V), -1)
    tips = []
    for f, (pers, root, mem) in enumerate(events[:n]):
        tips.append(root)
    # order fingers by x of the tip (thumb first for this model)
    for f, k in enumerate(sorted(range(n), key=lambda k: V[events[k][1], 0])):
        lab[events[k][2]] = f
    return lab, g, tips, [e[0] for e in events[:n + 2]]

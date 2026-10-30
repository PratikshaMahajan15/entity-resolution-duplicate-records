from collections import defaultdict
from itertools import combinations

import numpy as np


def true_pairs(entity_ids):
    groups = defaultdict(list)
    for pos, e in enumerate(entity_ids):
        groups[e].append(pos)
    return {p for idx in groups.values() if len(idx) > 1 for p in combinations(idx, 2)}


def pr_curve(scores, y, n_truth):
    """Precision/recall over thresholds. Recall is measured against ALL true pairs
    (n_truth), so pairs lost in blocking count as misses."""
    scores, y = np.asarray(scores), np.asarray(y)
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    precision = tp / k
    recall = tp / n_truth if n_truth else np.zeros(len(y))
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return scores[order], precision, recall, f1


def best_threshold(scores, y, n_truth):
    s, _, _, f1 = pr_curve(scores, y, n_truth)
    i = int(np.argmax(f1))
    return float(s[i]), float(f1[i])


def pair_metrics(pred, truth):
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": len(pred) - tp, "fn": len(truth) - tp}


def cluster_pairs(pred_pairs):
    """Transitive closure: connected components of the match graph -> all pairs inside."""
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(pred_pairs)
    closed = set()
    for comp in nx.connected_components(g):
        closed.update(combinations(sorted(comp), 2))
    return closed, nx.number_connected_components(g)

"""Competition metric, written by hand."""
import numpy as np


def roc_auc(y_true, scores):
    """AUC = the chance a random positive scores higher than a random negative."""
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)   # rank 1 = lowest score

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

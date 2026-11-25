"""Continual-learning metrics from an accuracy matrix.

A[k, j] = test accuracy on domain j after stage k.
Stage 0 = source model (before any adaptation). Domain j is first trained on
at stage j (domain 0 = source). Shape (K+1, K+1).
"""
import numpy as np


def summarize(A):
    A = np.asarray(A, dtype=float)
    K = A.shape[0] - 1
    out = {
        # average accuracy over all domains after the last stage (A_K)
        "acc_final": A[K].mean(),
        # mean over stages of the average accuracy on domains seen so far
        "acc_stream": np.mean([A[k, :k + 1].mean() for k in range(1, K + 1)]),
        # plasticity: accuracy on each target right after adapting to it
        "new_acc": np.mean([A[k, k] for k in range(1, K + 1)]),
        # forgetting: best earlier accuracy minus final accuracy (incl. source)
        "forgetting": np.mean([A[j:K, j].max() - A[K, j] for j in range(K)]),
        "src_final": A[K, 0],
        "tgt_final": A[K, 1:].mean(),
        # gain on the source domain's accuracy is negative forgetting of source
        "src_drop": A[0, 0] - A[K, 0],
    }
    if K >= 2:
        # forward transfer to not-yet-seen domains (relative to source model)
        out["fwd_unseen"] = np.mean([A[k, j] - A[0, j] for k in range(1, K) for j in range(k + 1, K + 1)])
        # gain on the LAST domain just before it is adapted to
        out["last_unseen_gain"] = A[K - 1, K] - A[0, K]
    # adaptation gain on each target vs the fixed source model
    out["adapt_gain"] = np.mean([A[k, k] - A[0, k] for k in range(1, K + 1)])
    return {k: float(v) for k, v in out.items()}

"""Byte-budgeted replay memory.

Budget definition (relative, dataset-agnostic):
    B = rho * (number of images seen so far) * bytes(one 224x224x3 uint8 image)
i.e. rho is the fraction of the seen stream that could be kept at full
resolution. Every storage format gets the same B bytes; lower-resolution
formats therefore hold more images (e.g. 64px holds ~12x more than 224px).

The memory is split equally between the domains seen so far and is
class-balanced inside each domain (by true label for the source, by
pseudo-label for targets).
"""
import numpy as np
import torch
import torch.nn.functional as F

from data import SIZE, to_input

REF_BYTES = 3 * SIZE * SIZE


def balanced_select(labels, k, rng, priority=None):
    """Pick k indices, round-robin across classes; within a class prefer high priority."""
    labels = np.asarray(labels)
    n = len(labels)
    if k >= n:
        return np.arange(n)
    if k <= 0:
        return np.array([], dtype=int)
    pr = rng.random(n) if priority is None else np.asarray(priority, dtype=float)
    order = np.lexsort((-pr, labels))                 # by class, then priority desc
    sl = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sl)) + 1]
    lengths = np.diff(np.r_[starts, n])
    rank_sorted = np.arange(n) - np.repeat(starts, lengths)
    ranks = np.empty(n, dtype=int)
    ranks[order] = rank_sorted
    sel = np.lexsort((rng.random(n), ranks))[:k]       # round-robin over classes
    return np.sort(sel)


class ReplayMemory:
    def __init__(self, res, rho, seed):
        self.res = int(res)
        self.rho = float(rho)
        self.item_bytes = 3 * self.res * self.res
        self.seen = 0
        self.store = {}          # domain code -> dict(x uint8, y, ytrue)
        self.rng = np.random.default_rng(seed + 12345)
        self._cat = None

    # ---- budget -----------------------------------------------------------
    def budget_bytes(self):
        return self.rho * self.seen * REF_BYTES

    def capacity(self):
        return int(self.budget_bytes() // self.item_bytes)

    def __len__(self):
        return sum(len(v["y"]) for v in self.store.values())

    def bytes_used(self):
        return len(self) * self.item_bytes

    # ---- write ------------------------------------------------------------
    def _encode(self, x224):
        if self.res == SIZE:
            return x224.clone()
        out = []
        for i in range(0, len(x224), 512):
            f = F.interpolate(x224[i:i + 512].float(), size=(self.res, self.res),
                              mode="bilinear", antialias=True, align_corners=False)
            out.append(f.round().clamp(0, 255).to(torch.uint8))
        return torch.cat(out)

    def add_domain(self, code, x224, labels, n_seen, priority=None, true_labels=None):
        """Insert samples from a finished domain and rebalance to the new budget."""
        self.seen += int(n_seen)
        quota = self.capacity() // (len(self.store) + 1)
        for c, v in list(self.store.items()):
            if len(v["y"]) > quota:
                keep = torch.as_tensor(balanced_select(v["y"].cpu().numpy(), quota, self.rng),
                                       device=v["y"].device, dtype=torch.long)
                self.store[c] = {k: t[keep] for k, t in v.items()}
        pr = None if priority is None else priority.float().cpu().numpy()
        sel = balanced_select(labels.cpu().numpy(), quota, self.rng, pr)
        idx = torch.as_tensor(sel, device=labels.device, dtype=torch.long)
        yt = true_labels if true_labels is not None else labels
        self.store[code] = {
            "x": self._encode(x224[idx]) if len(idx) else x224[:0, :, :self.res, :self.res].clone(),
            "y": labels[idx].clone(),
            "ytrue": yt[idx].clone(),
        }
        self._cat = None

    # ---- read -------------------------------------------------------------
    def sample(self, n):
        if self._cat is None:
            xs = [v["x"] for v in self.store.values() if len(v["y"])]
            ys = [v["y"] for v in self.store.values() if len(v["y"])]
            self._cat = (torch.cat(xs), torch.cat(ys)) if xs else None
        if self._cat is None:
            return None, None
        x, y = self._cat
        i = torch.randint(len(y), (n,), device=y.device)
        return to_input(x[i]), y[i]

    def stats(self):
        per = {}
        for c, v in self.store.items():
            n = len(v["y"])
            per[c] = {"n": n, "label_acc": (v["y"] == v["ytrue"]).float().mean().item() if n else None}
        return {"n_items": len(self), "bytes_used": self.bytes_used(),
                "budget_bytes": self.budget_bytes(), "capacity": self.capacity(),
                "seen": self.seen, "per_domain": per}

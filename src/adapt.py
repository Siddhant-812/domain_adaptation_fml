"""One continual adaptation run: source -> T1 -> T2 -> T3.

Methods
  fixed  : no adaptation (source model evaluated after every stage)
  seq    : sequential pseudo-label adaptation, update only `scope` params
  replay : seq + replay batch from a byte-budgeted memory, concatenated
           with the target batch (so BatchNorm sees both)

Target labels are NEVER used for training. They are only used to (a) compute
test accuracy and (b) log pseudo-label accuracy as a diagnostic.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (ADAPT_DEFAULTS, DATASETS, amp_ctx, atomic_json_dump, ckpt_path,
                    fill_adapt_cfg, get_device, load_valid_json, run_id, run_path, set_seed)
from data import augment, load_domain, num_classes, to_input
from memory import ReplayMemory
from metrics import summarize
from models import build_model, configure_scope, evaluate, predict, set_train_mode


def parse():
    ap = argparse.ArgumentParser()
    for k, v in ADAPT_DEFAULTS.items():
        typ = {int: int, float: float}.get(type(v), str)
        if k in ("lr", "tau"):
            typ = float
        ap.add_argument(f"--{k}", type=typ, default=v)
    ap.add_argument("--force", action="store_true")
    a = vars(ap.parse_args())
    force = a.pop("force")
    for k in ("order", "lr", "tau"):
        if a[k] in ("None", None):
            a[k] = None
    return fill_adapt_cfg(**a), force


def info_max_loss(logits):
    p = torch.softmax(logits, 1)
    ent = -(p * torch.log(p + 1e-6)).sum(1).mean()
    pm = p.mean(0)
    div = -(pm * torch.log(pm + 1e-6)).sum()
    return ent - div


def main():
    cfg, force = parse()
    rid = run_id(cfg)
    out = run_path(rid)
    if load_valid_json(out) is not None and not force:
        print(f"[skip] {rid} already done")
        return
    print(f"run {rid}\ncfg {cfg}", flush=True)
    set_seed(cfg["seed"])
    dev = get_device()
    amp = amp_ctx(dev)
    t0 = time.time()

    ds = DATASETS[cfg["dataset"]]
    codes = [ds["source"]] + list(cfg["order"])
    doms = [load_domain(cfg["dataset"], c, dev) for c in codes]
    K = len(codes) - 1

    model = build_model(cfg["arch"], num_classes(cfg["dataset"]), pretrained=False).to(dev)
    ck = torch.load(ckpt_path(cfg["dataset"], cfg["arch"], cfg["seed"]), map_location=dev)
    model.load_state_dict(ck["state_dict"])

    A = np.zeros((K + 1, K + 1))
    S = np.zeros((K + 1, K + 1))  # collapse index (max predicted-class share)

    def eval_row(k):
        for j, d in enumerate(doms):
            A[k, j], S[k, j] = evaluate(model, d, amp)
        print(f"stage {k} acc " + " ".join(f"{codes[j]}={A[k, j]:.3f}" for j in range(K + 1)), flush=True)

    eval_row(0)
    replay = cfg["method"] == "replay"
    mem = ReplayMemory(cfg["res"], cfg["rho"], cfg["seed"]) if replay else None
    if replay and cfg["mem"] == "srctgt":
        src = doms[0]
        tr = src["train_idx"]
        mem.add_domain(codes[0], src["x"][tr], src["y"][tr], n_seen=len(tr))
    stage_stats = []

    for k in range(1, K + 1):
        dom = doms[k]
        tr = dom["train_idx"]
        x_all, y_all = dom["x"], dom["y"]
        st = {"domain": codes[k], "epochs": []}
        if cfg["method"] != "fixed":
            params = configure_scope(model, cfg["scope"])
            opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9,
                                  weight_decay=0.0 if cfg["scope"] == "bn" else 5e-4)
            for ep in range(cfg["epochs"]):
                probs = predict(model, x_all, tr, amp)
                conf, pl = probs.max(1)
                mask = conf >= cfg["tau"]
                ytr = y_all[tr]
                st["epochs"].append({
                    "coverage": mask.float().mean().item(),
                    "pl_acc": (pl[mask] == ytr[mask]).float().mean().item() if mask.any() else None,
                    "pl_acc_all": (pl == ytr).float().mean().item(),
                })
                perm = torch.randperm(len(tr), device=dev)
                for i in range(0, len(perm), cfg["batch"]):
                    bi = perm[i:i + cfg["batch"]]
                    if len(bi) < 2:
                        continue
                    xb = augment(to_input(x_all[tr[bi]]))
                    parts = [xb]
                    yr = None
                    if replay:
                        xr, yr = mem.sample(cfg["batch"])
                        if xr is not None:
                            parts.append(augment(xr))
                    set_train_mode(model, cfg["scope"])
                    with amp:
                        logits = model(torch.cat(parts)).float()
                    lt = logits[:len(bi)]
                    loss = logits.new_zeros(())
                    m = mask[bi]
                    if m.any() and cfg["pl_weight"] > 0:
                        loss = loss + cfg["pl_weight"] * F.cross_entropy(lt[m], pl[bi][m])
                    if cfg["im_weight"] > 0:
                        loss = loss + cfg["im_weight"] * info_max_loss(lt)
                    if len(parts) > 1:
                        loss = loss + cfg["replay_weight"] * F.cross_entropy(logits[len(bi):], yr)
                    if loss.requires_grad:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
            e = st["epochs"][-1]
            print(f"  [{codes[k]}] last epoch coverage {e['coverage']:.2f} pl_acc {e['pl_acc']}", flush=True)
        eval_row(k)
        if replay:
            probs = predict(model, x_all, tr, amp)
            conf, pl = probs.max(1)
            mem.add_domain(codes[k], x_all[tr], pl, n_seen=len(tr), priority=conf, true_labels=y_all[tr])
            st["memory"] = mem.stats()
            print(f"  memory: {st['memory']['n_items']} items, "
                  f"{st['memory']['bytes_used'] / 1e6:.2f}/{st['memory']['budget_bytes'] / 1e6:.2f} MB", flush=True)
        stage_stats.append(st)

    summ = summarize(A)
    print("summary " + " ".join(f"{k}={v:.4f}" for k, v in summ.items()), flush=True)
    atomic_json_dump({
        "run_id": rid, "cfg": cfg, "domains": codes,
        "acc_matrix": A.tolist(), "maxshare_matrix": S.tolist(),
        "summary": summ, "stage_stats": stage_stats,
        "source_test_acc_ckpt": ck.get("source_test_acc"),
        "sec": time.time() - t0, "device": str(dev), "finished": True,
    }, out)
    print(f"saved {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

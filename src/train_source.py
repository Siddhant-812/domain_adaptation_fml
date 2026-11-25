"""Train the source model once per (dataset, arch, seed).

Every adaptation method for that seed starts from this SAME checkpoint, so
sequential vs replay comparisons are paired.
"""
import argparse
import os
import time

import torch
import torch.nn as nn

from common import DATASETS, amp_ctx, atomic_json_dump, ckpt_path, get_device, set_seed
from data import augment, load_domain, num_classes, to_input
from models import build_model, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="backbone lr; head uses 10x")
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--no_pretrained", action="store_true", help="smoke test only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = ckpt_path(a.dataset, a.arch, a.seed)
    if os.path.exists(out) and not a.force:
        print(f"[skip] {out} exists")
        return
    set_seed(a.seed)
    dev = get_device()
    amp = amp_ctx(dev)
    src = load_domain(a.dataset, DATASETS[a.dataset]["source"], dev)
    model = build_model(a.arch, num_classes(a.dataset), pretrained=not a.no_pretrained).to(dev)

    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.SGD([{"params": body, "lr": a.lr}, {"params": head, "lr": a.lr * 10}],
                          momentum=0.9, weight_decay=a.wd, nesterov=True)
    tr = src["train_idx"]
    steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=tr.device)]
        tot, n = 0.0, 0
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            if len(b) < 2:
                continue
            xb = augment(to_input(src["x"][b]))
            with amp:
                loss = crit(model(xb).float(), src["y"][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            n += len(b)
        if ep == a.epochs - 1 or ep % 5 == 0:
            acc, _ = evaluate(model, src, amp)
            print(f"epoch {ep:3d} loss {tot / max(n, 1):.4f} source-test acc {acc:.4f}", flush=True)

    acc, _ = evaluate(model, src, amp)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": a.arch, "dataset": a.dataset,
                "seed": a.seed, "source_test_acc": acc}, out + ".tmp")
    os.replace(out + ".tmp", out)
    atomic_json_dump({"source_test_acc": acc, "args": vars(a), "sec": time.time() - t0, "finished": True},
                     out.replace(".pt", ".json"))
    print(f"saved {out}  source test acc {acc:.4f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

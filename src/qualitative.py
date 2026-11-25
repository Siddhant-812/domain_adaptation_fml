"""Re-run one adaptation config and save the final model's test predictions on every domain.

adapt.py does not keep the adapted model, so this wrapper captures it (in place) and dumps
predictions for the qualitative figure. Use a RESULTS_DIR separate from results/.
Usage:  RESULTS_DIR=... python qualitative.py <name> <adapt.py args>
        python qualitative.py source --dataset pacs          (source checkpoint, no adaptation)
"""
import sys

import torch

import adapt
from common import DATASETS, amp_ctx, ckpt_path, get_device
from data import load_domain, num_classes
from models import build_model, predict

name = sys.argv[1]
holder = {}
if name == "source":
    ds = sys.argv[sys.argv.index("--dataset") + 1]
    dev = get_device()
    model = build_model("resnet18", num_classes(ds), pretrained=False).to(dev)
    model.load_state_dict(torch.load(ckpt_path(ds, "resnet18", 0), map_location=dev)["state_dict"])
    order = {"pacs": "ACS", "officehome": "ACP"}[ds]
else:
    sys.argv = [sys.argv[0]] + sys.argv[2:] + ["--force"]
    orig = adapt.evaluate

    def capture(model, dom, amp):
        holder["m"] = model
        return orig(model, dom, amp)

    adapt.evaluate = capture
    adapt.main()
    model = holder["m"]
    ds = sys.argv[sys.argv.index("--dataset") + 1]
    order = sys.argv[sys.argv.index("--order") + 1]
    dev = get_device()

out = {}
for code in [DATASETS[ds]["source"]] + list(order):
    d = load_domain(ds, code, dev)
    idx = d["test_idx"]
    p = predict(model, d["x"], idx, amp_ctx(dev))
    conf, pred = p.max(1)
    out[code] = {"test_idx": idx.cpu(), "pred": pred.cpu(), "conf": conf.cpu(), "y": d["y"][idx].cpu()}
    print(code, "test acc", (pred.cpu() == out[code]["y"]).float().mean().item())
import os
dst = os.path.join(os.environ["RESULTS_DIR"], f"qual_{ds}_{name}.pt")
os.makedirs(os.path.dirname(dst), exist_ok=True)
torch.save(out, dst)
print("saved", dst)

"""Shared configuration, defaults and small utilities."""
import contextlib
import json
import os
import random

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Datasets. Each domain has a one-letter code; folder = sub-directory of the
# dataset root that contains one sub-folder per class.
# ---------------------------------------------------------------------------
DATASETS = {
    "pacs": {
        "domains": {"P": "photo", "A": "art_painting", "C": "cartoon", "S": "sketch"},
        "source": "P",
        "default_order": "ACS",
        "root_env": "DATA_ROOT_PACS",
        "tau": 0.8,
    },
    "officehome": {
        "domains": {"R": "Real World", "A": "Art", "C": "Clipart", "P": "Product"},
        "source": "R",
        "default_order": "ACP",
        "root_env": "DATA_ROOT_OFFICEHOME",
        "tau": 0.6,  # 65 classes -> softmax confidences are lower
    },
    # synthetic dataset used only by the smoke test
    "fake": {
        "domains": {"P": "d0", "A": "d1", "C": "d2", "S": "d3"},
        "source": "P",
        "default_order": "ACS",
        "root_env": "DATA_ROOT_FAKE",
        "tau": 0.5,
    },
}

# Default learning rate per update scope (SGD, momentum 0.9).
SCOPE_LR = {"full": 1e-4, "lastblock": 1e-4, "bn": 1e-3, "head": 1e-3, "none": 0.0}

# Defaults for one adaptation run. make_jobs.py and adapt.py both use these so
# that run ids are computed identically everywhere.
ADAPT_DEFAULTS = {
    "dataset": "pacs",
    "arch": "resnet18",
    "seed": 0,
    "order": None,        # None -> dataset default order
    "method": "seq",      # fixed | seq | replay
    "scope": "bn",        # full | lastblock | bn | head | none
    "res": 64,            # replay image resolution (224, 112, 64, 32)
    "rho": 0.02,          # memory budget = rho * bytes(seen data at 224px)
    "mem": "srctgt",      # srctgt (source + past targets) | tgt (source-free)
    "epochs": 5,
    "batch": 64,
    "lr": None,           # None -> SCOPE_LR[scope]
    "tau": None,          # None -> dataset default
    "pl_weight": 1.0,
    "im_weight": 0.0,     # SHOT-style information-maximisation weight
    "replay_weight": 1.0,
}


def fill_adapt_cfg(**overrides):
    cfg = dict(ADAPT_DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    cfg.update({k: v for k, v in overrides.items()})
    ds = DATASETS[cfg["dataset"]]
    if cfg["order"] is None:
        cfg["order"] = ds["default_order"]
    if cfg["method"] == "fixed":
        cfg["scope"] = "none"
    if cfg["lr"] is None:
        cfg["lr"] = SCOPE_LR[cfg["scope"]]
    if cfg["tau"] is None:
        cfg["tau"] = ds["tau"]
    return cfg


def _fmt(v):
    return ("%g" % v) if isinstance(v, float) else str(v)


def run_id(cfg):
    """Deterministic, human-readable id. Only non-default knobs are appended."""
    c = cfg
    rid = f"{c['dataset']}_{c['arch']}_s{c['seed']}_{c['order']}_{c['method']}_{c['scope']}"
    if c["method"] == "replay":
        rid += f"_r{c['res']}_rho{_fmt(c['rho'])}_{c['mem']}"
    if c["im_weight"] > 0:
        rid += f"_im{_fmt(c['im_weight'])}_pl{_fmt(c['pl_weight'])}"
    elif c["pl_weight"] != ADAPT_DEFAULTS["pl_weight"]:
        rid += f"_pl{_fmt(c['pl_weight'])}"
    if c["lr"] != SCOPE_LR[c["scope"]]:
        rid += f"_lr{_fmt(c['lr'])}"
    if c["tau"] != DATASETS[c["dataset"]]["tau"]:
        rid += f"_tau{_fmt(c['tau'])}"
    for k in ("epochs", "batch", "replay_weight"):
        if c[k] != ADAPT_DEFAULTS[k]:
            rid += f"_{k}{_fmt(c[k])}"
    return rid


def cfg_to_cli(cfg):
    """Turn a config dict into CLI args for adapt.py (no spaces in values)."""
    parts = []
    for k, v in cfg.items():
        parts.append(f"--{k} {_fmt(v)}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Paths (all come from env.sh)
# ---------------------------------------------------------------------------
def env_path(name, default=None):
    v = os.environ.get(name, default)
    if v is None:
        raise SystemExit(f"Environment variable {name} is not set. Edit and source env.sh.")
    return v


def cache_dir():
    return env_path("CACHE_DIR")


def ckpt_dir():
    return env_path("CKPT_DIR")


def results_dir():
    return env_path("RESULTS_DIR")


def ckpt_path(dataset, arch, seed):
    return os.path.join(ckpt_dir(), f"{dataset}_{arch}_s{seed}.pt")


def run_path(rid):
    return os.path.join(results_dir(), "runs", f"{rid}.json")


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_ctx(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def atomic_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_valid_json(path):
    """Return parsed result if it exists, is non-empty and marked finished."""
    try:
        if os.path.getsize(path) == 0:
            return None
        with open(path) as f:
            obj = json.load(f)
        return obj if obj.get("finished") else None
    except (OSError, json.JSONDecodeError):
        return None

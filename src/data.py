"""Dataset caching (images -> uint8 tensors) and GPU-side preprocessing.

Run once per dataset:
    python src/data.py --dataset pacs
This writes one file per domain to $CACHE_DIR/<dataset>/<code>.pt containing
uint8 images (N,3,224,224), labels and a fixed stratified 80/20 split
(train = unlabeled adaptation pool for targets / training set for source,
 test = held-out evaluation set). The split never changes across seeds.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import DATASETS, cache_dir, env_path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _load_img(path, size=SIZE, resize=256):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = resize / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    im = im.crop((left, top, left + size, top + size))
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).copy()


def stratified_split(labels, test_frac, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train, test = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.sort(train), np.sort(test)


def domain_cache_path(dataset, code):
    return os.path.join(cache_dir(), dataset, f"{code}.pt")


def prepare_dataset(dataset, root, test_frac=0.2, split_seed=0, workers=8, force=False):
    info = DATASETS[dataset]
    classes = None
    for code, folder in info["domains"].items():
        dpath = os.path.join(root, folder)
        if not os.path.isdir(dpath):
            raise SystemExit(f"Missing domain folder: {dpath}")
        cls = sorted(d for d in os.listdir(dpath) if os.path.isdir(os.path.join(dpath, d)))
        if classes is None:
            classes = cls
        elif cls != classes:
            missing = set(classes) ^ set(cls)
            raise SystemExit(f"Class folders differ in {dpath}: {sorted(missing)[:10]}")
        out = domain_cache_path(dataset, code)
        if os.path.exists(out) and not force:
            print(f"[skip] {out} exists")
            continue
        files, labels = [], []
        for ci, c in enumerate(classes):
            cdir = os.path.join(dpath, c)
            for fn in sorted(os.listdir(cdir)):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    files.append(os.path.join(cdir, fn))
                    labels.append(ci)
        print(f"[{dataset}/{code}] {folder}: {len(files)} images, {len(classes)} classes")
        with ThreadPoolExecutor(workers) as ex:
            arrs = list(ex.map(_load_img, files))
        x = torch.from_numpy(np.stack(arrs))
        y = torch.tensor(labels, dtype=torch.long)
        tr, te = stratified_split(labels, test_frac, split_seed)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        torch.save({"x": x, "y": y, "train_idx": torch.from_numpy(tr), "test_idx": torch.from_numpy(te),
                    "classes": classes, "folder": folder}, tmp)
        os.replace(tmp, out)
        print(f"  saved {out}  train={len(tr)} test={len(te)}")


def load_domain(dataset, code, device):
    d = torch.load(domain_cache_path(dataset, code), map_location="cpu")
    for k in ("x", "y", "train_idx", "test_idx"):
        d[k] = d[k].to(device)
    return d


def num_classes(dataset):
    src = DATASETS[dataset]["source"]
    return len(torch.load(domain_cache_path(dataset, src), map_location="cpu")["classes"])


# ---------------------------------------------------------------------------
# GPU preprocessing
# ---------------------------------------------------------------------------
def to_input(x_uint8, size=SIZE):
    """uint8 (B,3,h,w) -> normalised float (B,3,size,size). Upsamples low-res memory images."""
    x = x_uint8.float().div_(255.0)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def augment(x):
    """Random resized crop (scale 0.75-1) + horizontal flip, vectorised on device."""
    b, dev = x.shape[0], x.device
    scale = torch.empty(b, device=dev).uniform_(0.75, 1.0)
    flip = (torch.rand(b, device=dev) < 0.5).float() * 2 - 1
    tx = (torch.rand(b, device=dev) * 2 - 1) * (1 - scale)
    ty = (torch.rand(b, device=dev) * 2 - 1) * (1 - scale)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = scale * flip
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--root", default=None, help="defaults to $DATA_ROOT_<DATASET>")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = a.root or env_path(DATASETS[a.dataset]["root_env"])
    prepare_dataset(a.dataset, root, workers=a.workers, force=a.force)

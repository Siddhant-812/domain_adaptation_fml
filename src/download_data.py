"""Download PACS and Office-Home and write them as <out>/<domain>/<class>/<img>.

    python src/download_data.py --dataset pacs       --out $DATA_ROOT_PACS
    python src/download_data.py --dataset officehome --out $DATA_ROOT_OFFICEHOME

Source: Hugging Face mirrors maintained by Flower Labs (parquet, original image
bytes, full datasets):
  PACS        flwrlabs/pacs         9,991 images, 7 classes, 4 domains
  Office-Home flwrlabs/office-home  15,588 images, 65 classes, 4 domains
Image bytes are written as-is (no re-encoding). Run on a node with internet
access (usually the login node). Needs `pip install datasets`.

If Hugging Face is blocked, use the official pages instead and unpack so that the
same layout results:
  PACS        https://sketchx.eecs.qmul.ac.uk/downloads/   (folder: kfold/)
  Office-Home https://www.hemanthdv.org/officeHomeDataset.html
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS  # noqa: E402

HF_REPO = {"pacs": "flwrlabs/pacs", "officehome": "flwrlabs/office-home"}


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def ext_from_bytes(b):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:2] == b"\xff\xd8":
        return ".jpg"
    if b[:2] == b"BM":
        return ".bmp"
    return ".jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(HF_REPO))
    ap.add_argument("--out", required=True, help="dataset root (= DATA_ROOT_* in env.sh)")
    a = ap.parse_args()

    from datasets import Image, load_dataset

    folders = list(DATASETS[a.dataset]["domains"].values())
    by_norm = {norm(f): f for f in folders}
    by_norm.update({"art": "art_painting"} if a.dataset == "pacs" else {})

    print(f"downloading {HF_REPO[a.dataset]} ...", flush=True)
    ds = load_dataset(HF_REPO[a.dataset], split="train")
    classes = ds.features["label"].names
    domains = sorted(set(ds["domain"]))
    mapping = {}
    for d in domains:
        if norm(d) not in by_norm:
            raise SystemExit(f"Unknown domain '{d}' in {HF_REPO[a.dataset]}; expected {folders}. "
                             f"Edit the domain mapping in this script.")
        mapping[d] = by_norm[norm(d)]
    print(f"domains: {mapping}\nclasses: {len(classes)}", flush=True)

    for f in folders:
        for c in classes:
            os.makedirs(os.path.join(a.out, f, c), exist_ok=True)

    ds = ds.cast_column("image", Image(decode=False))
    counts = {}
    for i, ex in enumerate(ds):
        b = ex["image"]["bytes"]
        folder = mapping[ex["domain"]]
        cls = classes[ex["label"]]
        path = os.path.join(a.out, folder, cls, f"{i:06d}{ext_from_bytes(b)}")
        if not os.path.exists(path):
            with open(path, "wb") as fh:
                fh.write(b)
        counts[folder] = counts.get(folder, 0) + 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)
    print("images per domain:", counts)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()

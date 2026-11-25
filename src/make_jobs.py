"""Define all experiments and write SLURM job lists.

Writes:
  jobs/source.txt    one line per source checkpoint to train
  jobs/adapt.txt     one line per (deduplicated) adaptation run
  jobs/manifest.json experiment group -> list of run ids (used by analyze.py)

Each line is the argument string after `python`, e.g.
  src/adapt.py --dataset pacs --seed 0 ...

Experiment design (sized for a 4-5 page report, 36 short runs, one seed):
  E1 scope    : does restricting the update scope prevent collapse?
  E2 main     : fixed vs sequential vs replay, 2 datasets
  E3 memory   : what to store at a given byte budget (resolution x budget)
  E4 ablation : source-free memory; domain order / transfer
"""
import argparse
import json
import os

SEEDS = [0]

from common import ckpt_path, cfg_to_cli, fill_adapt_cfg, load_valid_json, run_id, run_path


def experiments(smoke=False):
    if smoke:
        datasets = ["fake"]
        e3_res, e3_rho = [224, 64, 32], [0.05, 0.2]
        main_rho, extra = 0.1, {"epochs": 1}
        orders = ["CAS"]
        e3_ds = "fake"
    else:
        datasets = ["pacs", "officehome"]
        e3_res, e3_rho = [224, 112, 64, 32], [0.005, 0.01, 0.02, 0.05]
        main_rho, extra = 0.02, {}
        orders = ["CAS", "SCA"]
        e3_ds = "pacs"

    seeds3 = seeds5 = SEEDS                # one seed for every experiment
    G = {}

    def add(group, **kw):
        G.setdefault(group, []).append(fill_adapt_cfg(**kw, **extra))

    # E1: update scope (sequential pseudo-labelling only)
    for ds in datasets:
        for s in seeds3:
            for scope in ["full", "lastblock", "bn", "head"]:
                add("E1_scope", dataset=ds, seed=s, method="seq", scope=scope)
            # SHOT-style information maximisation on the full encoder
            add("E1_scope", dataset=ds, seed=s, method="seq", scope="full", im_weight=1.0, pl_weight=0.3)

    # E2: main paired comparison
    for ds in datasets:
        for s in seeds5:
            add("E2_main", dataset=ds, seed=s, method="fixed")
            for scope in ["head", "bn"]:
                add("E2_main", dataset=ds, seed=s, method="seq", scope=scope)
                add("E2_main", dataset=ds, seed=s, method="replay", scope=scope, res=64, rho=main_rho)

    # E3: memory format x budget (PACS, BN-affine scope)
    for s in seeds3:
        for res in e3_res:
            for rho in e3_rho:
                add("E3_memory", dataset=e3_ds, seed=s, method="replay", scope="bn", res=res, rho=rho)

    # E4a: source-free memory (only past target samples, pseudo-labelled)
    for s in seeds5:
        add("E4_sourcefree", dataset=e3_ds, seed=s, method="replay", scope="bn", res=64, rho=main_rho, mem="tgt")
        add("E4_sourcefree", dataset=e3_ds, seed=s, method="replay", scope="bn", res=64, rho=main_rho)
        add("E4_sourcefree", dataset=e3_ds, seed=s, method="seq", scope="bn")

    # E4b: domain order (transfer to unseen domains)
    for s in seeds3:
        for order in [None] + orders:
            add("E4_order", dataset=e3_ds, seed=s, order=order, method="seq", scope="bn")
            add("E4_order", dataset=e3_ds, seed=s, order=order, method="replay", scope="bn", res=64, rho=main_rho)
    return G, datasets, SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny config on the fake dataset")
    ap.add_argument("--only-missing", action="store_true", help="skip runs with a valid result file")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--outdir", default="jobs")
    a = ap.parse_args()

    G, datasets, seeds = experiments(a.smoke)
    os.makedirs(a.outdir, exist_ok=True)
    src_extra = " --epochs 1 --no_pretrained" if a.smoke else ""

    src_lines = []
    for ds in datasets:
        for s in seeds:
            if a.only_missing and os.path.exists(ckpt_path(ds, a.arch, s)):
                continue
            src_lines.append(f"src/train_source.py --dataset {ds} --arch {a.arch} --seed {s}{src_extra}")

    manifest, seen, adapt_lines = {}, set(), []
    for g, cfgs in G.items():
        manifest[g] = []
        for c in cfgs:
            c["arch"] = a.arch
            rid = run_id(c)
            manifest[g].append(rid)
            if rid in seen:
                continue
            seen.add(rid)
            if a.only_missing and load_valid_json(run_path(rid)) is not None:
                continue
            adapt_lines.append(f"src/adapt.py {cfg_to_cli(c)}")

    with open(os.path.join(a.outdir, "source.txt"), "w") as f:
        f.write("".join(l + "\n" for l in src_lines))
    with open(os.path.join(a.outdir, "adapt.txt"), "w") as f:
        f.write("".join(l + "\n" for l in adapt_lines))
    with open(os.path.join(a.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"source jobs: {len(src_lines)}   adapt jobs: {len(adapt_lines)} "
          f"(unique runs total: {len(seen)})")
    for g, r in manifest.items():
        print(f"  {g}: {len(r)} runs")


if __name__ == "__main__":
    main()

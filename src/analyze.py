"""Collect results and produce every table and figure for the report.

    python src/analyze.py [--manifest jobs/manifest.json] [--out analysis]

Outputs (in --out):
  run_status.csv             every expected run and whether its result is valid
  all_runs.csv               one row per finished run (config + metrics)
  tables.md / tables.tex     all report tables
  fig_*.pdf / fig_*.png      all report figures
  summary.md                 status + tables in one file (send this back)
"""
import argparse
import json
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, load_valid_json, run_path  # noqa: E402

DS_NAME = {"pacs": "PACS", "officehome": "Office-Home", "fake": "Fake"}
SCOPE_NAME = {"full": "Full encoder", "lastblock": "Last block", "bn": "BN-affine",
              "head": "Head only", "none": "-"}
METRICS = ["acc_final", "forgetting", "new_acc", "src_final", "tgt_final"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def pct(x):
    return 100 * x


def ms(series, digits=1):
    """Value in % (mean if several runs match)."""
    v = pct(np.asarray(series, dtype=float))
    v = v[~np.isnan(v)]
    return f"{v.mean():.{digits}f}" if len(v) else "n/a"


def load(manifest):
    status, rows, raw = [], [], {}
    for g, rids in manifest.items():
        for rid in rids:
            r = load_valid_json(run_path(rid))
            status.append({"group": g, "run_id": rid, "ok": r is not None})
            if r is None or rid in raw:
                continue
            raw[rid] = r
            c = r["cfg"]
            row = {**c, "run_id": rid, **r["summary"]}
            S = np.array(r["maxshare_matrix"])
            row["collapse_final"] = S[-1, 1:].mean()
            pls = [st["epochs"][-1] for st in r["stage_stats"] if st["epochs"]]
            row["pl_acc_last"] = np.mean([e["pl_acc"] for e in pls if e["pl_acc"] is not None]) if pls else np.nan
            row["coverage_last"] = np.mean([e["coverage"] for e in pls]) if pls else np.nan
            mems = [st["memory"] for st in r["stage_stats"] if "memory" in st]
            row["mem_items"] = mems[-1]["n_items"] if mems else 0
            row["mem_MB"] = mems[-1]["bytes_used"] / 1e6 if mems else 0
            row["sec"] = r["sec"]
            rows.append(row)
    return pd.DataFrame(status), pd.DataFrame(rows), raw


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def tex_table(df, caption):
    body = df.to_latex(index=False, escape=True, column_format="l" * len(df.columns))
    body = body.replace("Δ", r"$\Delta$").replace("ρ", r"$\rho$").replace("→", r"$\to$")
    return "% " + caption + "\n" + body


# ---------------------------------------------------------------------------
def table_e1(df):
    d = df[df.run_id.isin(GROUPS.get("E1_scope", []))].copy()
    if d.empty:
        return None
    d["variant"] = d.apply(lambda r: SCOPE_NAME[r.scope] + (" + IM" if r.im_weight > 0 else ""), axis=1)
    order = ["Full encoder", "Full encoder + IM", "Last block", "BN-affine", "Head only"]
    out = []
    for ds in [x for x in DS_NAME if x in d.dataset.unique()]:
        fx = df[(df.dataset == ds) & (df.method == "fixed")]
        if len(fx):
            out.append({"Dataset": DS_NAME[ds], "Update scope": "Fixed source (no update)",
                        "ACC": ms(fx.acc_final), "Forgetting": ms(fx.forgetting),
                        "New-domain acc": ms(fx.new_acc), "PL acc": "-", "Collapse idx": ms(fx.collapse_final)})
        for v in order:
            g = d[(d.dataset == ds) & (d.variant == v)]
            if g.empty:
                continue
            out.append({"Dataset": DS_NAME[ds], "Update scope": v, "ACC": ms(g.acc_final),
                        "Forgetting": ms(g.forgetting), "New-domain acc": ms(g.new_acc),
                        "PL acc": ms(g.pl_acc_last), "Collapse idx": ms(g.collapse_final)})
    return pd.DataFrame(out)


def table_e2(df):
    d = df[df.run_id.isin(GROUPS.get("E2_main", []))]
    if d.empty:
        return None, None
    out, diffs = [], []
    for ds in [x for x in DS_NAME if x in d.dataset.unique()]:
        fx = d[(d.dataset == ds) & (d.method == "fixed")]
        out.append({"Dataset": DS_NAME[ds], "Scope": "-", "Method": "Fixed source",
                    **{m: ms(fx[m]) for m in METRICS}})
        for scope in ["head", "bn"]:
            for meth in ["seq", "replay"]:
                g = d[(d.dataset == ds) & (d.scope == scope) & (d.method == meth)]
                if g.empty:
                    continue
                out.append({"Dataset": DS_NAME[ds], "Scope": SCOPE_NAME[scope],
                            "Method": "Sequential" if meth == "seq" else "Replay",
                            **{m: ms(g[m]) for m in METRICS}})
            a = d[(d.dataset == ds) & (d.scope == scope) & (d.method == "seq")].set_index("seed")
            b = d[(d.dataset == ds) & (d.scope == scope) & (d.method == "replay")].set_index("seed")
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            row = {"Dataset": DS_NAME[ds], "Scope": SCOPE_NAME[scope]}
            for m, name in [("acc_final", "ΔACC"), ("forgetting", "ΔForgetting"), ("new_acc", "ΔNew-domain acc"),
                            ("src_final", "ΔSource acc")]:
                diff = pct(b.loc[common, m].values - a.loc[common, m].values).mean()
                row[name] = f"{diff:+.1f}"
            diffs.append(row)
    rename = {"acc_final": "ACC", "forgetting": "Forgetting", "new_acc": "New-domain acc",
              "src_final": "Source acc", "tgt_final": "Target acc"}
    return pd.DataFrame(out).rename(columns=rename), pd.DataFrame(diffs)


def table_e3(df):
    d = df[df.run_id.isin(GROUPS.get("E3_memory", []))]
    if d.empty:
        return None
    rhos = sorted(d.rho.unique())
    out = []
    for res in sorted(d.res.unique(), reverse=True):
        row = {"Format": f"raw{res}"}
        for rho in rhos:
            g = d[(d.res == res) & (d.rho == rho)]
            row[f"ρ={rho * 100:g}%"] = ms(g.acc_final) + f" (n={g.mem_items.mean():.0f})" if len(g) else "n/a"
        out.append(row)
    seq = df[(df.dataset == d.dataset.iloc[0]) & (df.method == "seq") & (df.scope == "bn")
             & (df.order == DATASETS[d.dataset.iloc[0]]["default_order"])]
    if len(seq):
        out.append({"Format": "no replay (seq)", **{f"ρ={r * 100:g}%": ms(seq.acc_final) for r in rhos}})
    return pd.DataFrame(out)


def table_e4(df):
    out = []
    d = df[df.run_id.isin(GROUPS.get("E4_sourcefree", []))]
    for (meth, mem), g in d.groupby(["method", "mem"]):
        label = "Sequential (no memory)" if meth == "seq" else (
            "Replay, source + past targets" if mem == "srctgt" else "Replay, past targets only (source-free)")
        out.append({"Study": "Memory content", "Setting": label, "ACC": ms(g.acc_final),
                    "Forgetting": ms(g.forgetting), "Source acc": ms(g.src_final),
                    "Unseen-domain gain": "-"})
    d = df[df.run_id.isin(GROUPS.get("E4_order", []))]
    for (order, meth), g in d.groupby(["order", "method"]):
        names = DATASETS[g.dataset.iloc[0]]["domains"]
        label = "→".join(names[c] for c in order) + (" (replay)" if meth == "replay" else " (seq)")
        out.append({"Study": "Domain order", "Setting": label, "ACC": ms(g.acc_final),
                    "Forgetting": ms(g.forgetting), "Source acc": ms(g.src_final),
                    "Unseen-domain gain": ms(g.last_unseen_gain)})
    return pd.DataFrame(out) if out else None


# ---------------------------------------------------------------------------
def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def fig_scope(df, out):
    d = df[df.run_id.isin(GROUPS.get("E1_scope", []))].copy()
    if d.empty:
        return
    d["variant"] = d.apply(lambda r: SCOPE_NAME[r.scope] + (" + IM" if r.im_weight > 0 else ""), axis=1)
    order = ["Full encoder", "Full encoder + IM", "Last block", "BN-affine", "Head only"]
    dss = [x for x in DS_NAME if x in d.dataset.unique()]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 2.6), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        vals = [pct(d[(d.dataset == ds) & (d.variant == v)].acc_final) for v in order]
        means = [v.mean() if len(v) else np.nan for v in vals]
        ax.bar(range(len(order)), means, color="#4C72B0")
        fx = df[(df.dataset == ds) & (df.method == "fixed")]
        if len(fx):
            ax.axhline(pct(fx.acc_final).mean(), ls="--", c="k", lw=1, label="fixed source")
            ax.legend(frameon=False, loc="upper left")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace(" + ", "\n+ ") for o in order], rotation=30, ha="right")
        ax.set_ylabel("Final average accuracy (%)")
        ax.set_title(DS_NAME[ds])
    save(fig, out, "fig1_scope")


def mean_matrix(raw, rids):
    Ms = [np.array(raw[r]["acc_matrix"]) for r in rids if r in raw]
    return (np.mean(Ms, 0), raw[[r for r in rids if r in raw][0]]["domains"]) if Ms else (None, None)


def fig_matrices(df, raw, out):
    d = df[df.run_id.isin(GROUPS.get("E2_main", []))]
    for ds in d.dataset.unique():
        panels = []
        for scope in ["head", "bn"]:
            for meth in ["seq", "replay"]:
                rids = d[(d.dataset == ds) & (d.scope == scope) & (d.method == meth)].run_id
                M, codes = mean_matrix(raw, list(rids))
                if M is not None:
                    panels.append((f"{SCOPE_NAME[scope]} / {'Sequential' if meth == 'seq' else 'Replay'}", M, codes))
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels), 2.5), squeeze=False)
        for ax, (title, M, codes) in zip(axes[0], panels):
            ax.imshow(pct(M), vmin=0, vmax=100, cmap="viridis")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{pct(M[i, j]):.0f}", ha="center", va="center",
                            color="w" if M[i, j] < 0.6 else "k", fontsize=8)
            ax.set_xticks(range(len(codes)))
            ax.set_xticklabels(codes)
            ax.set_yticks(range(len(codes)))
            ax.set_yticklabels(["src"] + [f"after {c}" for c in codes[1:]])
            ax.set_xlabel("evaluated on")
            ax.set_title(title, fontsize=8)
        save(fig, out, f"fig2_accmatrix_{ds}")


def fig_curves(df, raw, out):
    d = df[df.run_id.isin(GROUPS.get("E2_main", []))]
    for ds in d.dataset.unique():
        res = {}
        for meth in ["seq", "replay"]:
            rids = d[(d.dataset == ds) & (d.scope == "bn") & (d.method == meth)].run_id
            res[meth] = mean_matrix(raw, list(rids))
        if res["seq"][0] is None:
            continue
        codes = res["seq"][1]
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        colors = plt.cm.tab10.colors
        for j, c in enumerate(codes):
            for meth, ls in [("seq", "--"), ("replay", "-")]:
                M = res[meth][0]
                if M is None:
                    continue
                ax.plot(range(j, len(codes)), pct(M[j:, j]), ls, color=colors[j], marker="o", ms=3,
                        label=f"{c} ({'replay' if meth == 'replay' else 'seq'})")
        ax.set_xticks(range(len(codes)))
        ax.set_xticklabels(["source"] + [f"+{c}" for c in codes[1:]])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xlabel("training stage")
        ax.set_title(f"{DS_NAME[ds]}, BN-affine: dashed = sequential, solid = replay", fontsize=8)
        ax.legend(frameon=False, fontsize=6, ncol=2)
        save(fig, out, f"fig3_curves_{ds}")


def fig_memory(df, out):
    d = df[df.run_id.isin(GROUPS.get("E3_memory", []))]
    if d.empty:
        return
    ds = d.dataset.iloc[0]
    seq = df[(df.dataset == ds) & (df.method == "seq") & (df.scope == "bn")
             & (df.order == DATASETS[ds]["default_order"])]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    fig.subplots_adjust(wspace=0.35)
    for res in sorted(d.res.unique(), reverse=True):
        g = d[d.res == res].groupby("rho")
        x = pct(np.array(sorted(d[d.res == res].rho.unique())))
        for ax, m in zip(axes[:2], ["acc_final", "forgetting"]):
            ax.plot(x, pct(g[m].mean()), marker="o", ms=3, label=f"raw{res}")
        axes[2].plot(g.mem_items.mean(), pct(g.acc_final.mean()), marker="o", ms=3, label=f"raw{res}")
    for ax, m, yl in [(axes[0], "acc_final", "Final average accuracy (%)"),
                      (axes[1], "forgetting", "Forgetting (pp)")]:
        if len(seq):
            ax.axhline(pct(seq[m]).mean(), ls="--", c="k", lw=1, label="no replay")
        ax.set_xscale("log")
        ax.set_xlabel("budget ρ (% of seen data)")
        ax.set_xticks(pct(np.array(sorted(d.rho.unique()))))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.minorticks_off()
        ax.set_ylabel(yl)
    if len(seq):
        axes[2].axhline(pct(seq.acc_final).mean(), ls="--", c="k", lw=1, label="no replay")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("images stored at end of stream")
    axes[2].set_ylabel("Final average accuracy (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{DS_NAME[ds]}, BN-affine: equal bytes, different resolutions", fontsize=9)
    save(fig, out, "fig4_memory")


def fig_pl(raw, out):
    """Pseudo-label accuracy per stage for each scope (explains collapse)."""
    rids = [r for r in GROUPS.get("E1_scope", []) if r in raw]
    if not rids:
        return
    rows = []
    for r in rids:
        c = raw[r]["cfg"]
        v = SCOPE_NAME[c["scope"]] + (" + IM" if c["im_weight"] > 0 else "")
        for k, st in enumerate(raw[r]["stage_stats"]):
            for e, ep in enumerate(st["epochs"]):
                rows.append({"dataset": c["dataset"], "variant": v, "t": k * len(st["epochs"]) + e,
                             "pl_acc_all": ep["pl_acc_all"]})
    d = pd.DataFrame(rows)
    dss = [x for x in DS_NAME if x in d.dataset.unique()]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 2.4), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        for v, g in d[d.dataset == ds].groupby("variant"):
            m = g.groupby("t").pl_acc_all.mean()
            ax.plot(m.index, pct(m.values), label=v, lw=1.2)
        ax.set_xlabel("epoch (across all target stages)")
        ax.set_ylabel("Pseudo-label accuracy (%)")
        ax.set_title(DS_NAME[ds])
    axes[0][0].legend(frameon=False, fontsize=6)
    save(fig, out, "fig5_pseudolabels")


# ---------------------------------------------------------------------------
GROUPS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="jobs/manifest.json")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(a.manifest) as f:
        GROUPS.update(json.load(f))
    status, df, raw = load(GROUPS)
    status.to_csv(os.path.join(a.out, "run_status.csv"), index=False)
    if df.empty:
        print("No finished runs yet.")
        return
    df.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)

    t1 = table_e1(df)
    t2, t2d = table_e2(df)
    t3 = table_e3(df)
    t4 = table_e4(df)
    tables = [("Table 1. Update scope, sequential pseudo-labelling (%)", t1),
              ("Table 2. Main results (%)", t2),
              ("Table 2b. Effect of replay: Replay − Sequential (percentage points)", t2d),
              ("Table 3. Memory format × budget: final ACC in % (n = images stored)", t3),
              ("Table 4. Ablations (%)", t4)]
    md, tex = [], []
    for cap, t in tables:
        if t is None or len(t) == 0:
            continue
        md.append(f"### {cap}\n\n{md_table(t)}\n")
        tex.append(tex_table(t, cap))
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(a.out, "tables.tex"), "w") as f:
        f.write("\n\n".join(tex))

    fig_scope(df, a.out)
    fig_matrices(df, raw, a.out)
    fig_curves(df, raw, a.out)
    fig_memory(df, a.out)
    fig_pl(raw, a.out)

    n_ok = status.drop_duplicates("run_id").ok.sum()
    n_all = status.run_id.nunique()
    missing = status[~status.ok].drop_duplicates("run_id")
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(f"# Results summary\n\nFinished runs: {n_ok}/{n_all}\n\n")
        if len(missing):
            f.write("Missing/invalid runs:\n\n" + "\n".join(f"- {r}" for r in missing.run_id) + "\n\n")
        f.write(f"Total compute: {df.sec.sum() / 3600:.2f} GPU-hours\n\n")
        f.write("\n".join(md))
        f.write("\n\nFigures: " + ", ".join(sorted(x for x in os.listdir(a.out) if x.endswith(".png"))) + "\n")
    print(f"finished {n_ok}/{n_all} runs; wrote tables and figures to {a.out}/")
    if len(missing):
        print(f"WARNING: {len(missing)} runs missing or invalid (see run_status.csv)")


if __name__ == "__main__":
    main()

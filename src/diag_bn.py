"""Diagnostic wrapper around adapt.py (does not modify the study code or results/).

BN_DIAG=frozen_stats : BN affine params are trained, but BN layers stay in eval mode, so the
                       running statistics keep the source model's values.
BN_DIAG=stats_only   : normal BN-scope train mode (running stats update); pass `--lr 0` so no
                       parameter changes. Isolates the effect of statistics drift alone.
Run with RESULTS_DIR pointing at a separate directory.
"""
import os

import adapt

mode = os.environ.get("BN_DIAG", "")
if mode == "frozen_stats":
    adapt.set_train_mode = lambda model, scope: model.eval()
elif mode != "stats_only":
    raise SystemExit("set BN_DIAG=frozen_stats or BN_DIAG=stats_only")
adapt.main()

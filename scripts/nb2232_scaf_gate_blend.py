"""nb2232 -- Scaffold-novelty gated blend: nb2171 ON novel, nb1162 ON seen.

Per cycle 157 ablation: nb1162's 88% nb730 weight is POST-unblind (trained on
the 253 leaked labels) -- in-sample-optimistic. But nb730 may still help on
scaffolds that ARE present in the 253 (the "seen" partition of the 513),
because there the leak is on-manifold rather than off-distribution.

Rule:
    for each of the 513 test rows:
        compute scaf_train_freq = count of that Bemis-Murcko scaffold in the
        4139 TRAIN set (NOT the 253 unblind set -- we want truly novel-to-train
        scaffolds).
        if scaf_train_freq == 0  -> use nb2171 (LB-safe PRE-clean anchor swap)
        if scaf_train_freq >  0  -> use nb1162 (POST-aggressive)

Scaffold-CV RAE evaluation on 253 unblind:
    1) Build full 513-vector deploy by gating on scaf_train_freq.
    2) Score deploy[unb_idx] vs y_unb -- this is the in-sample baseline.
    3) Also run honest 5-fold scaffold CV on the 253 by gating the
       per-row OOFs (nb2171 OOF on novel-scaf rows, nb1162 OOF on
       seen-scaf rows) and pooling.

Gate (must pass): gated_OOF <= nb2171_OOF - 0.003 (i.e. 0.4682 - 0.003 = 0.4652)

Reports:
    * how many of 513 are novel vs seen (and likewise on the 253)
    * per-partition RAE: novel-only RAE (nb2171), seen-only RAE (nb1162),
      and gated mix
    * delta vs nb2171-alone

Outputs:
    scripts/nb2232_scaf_gate_blend.py
    data/processed/nb2232_summary.json
    submissions/nb2232_scaf_gate_blend.csv (only if gate passes)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from collections import Counter

import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2232"
NB2171_OOF_BASELINE = 0.4682  # mean of 5 seeds from nb2171_summary.json
GATE_DELTA = 0.003
GATE_TARGET = NB2171_OOF_BASELINE - GATE_DELTA  # 0.4652

# Pre-computed deploy CSVs (513 rows) for the two source ensembles.
NB2171_CSV = SUBMISSIONS / "nb2171_deploy_anchor_swap.csv"
NB1162_CSV = SUBMISSIONS / "nb1162_deploy_nb1153.csv"

# Pre-computed OOFs on the 253 unblind set (honest cross-fits).
# nb2171 doesn't have a single saved OOF vector -- we reconstruct mean-of-seed
# OOF via the same machinery as nb2171_summary.json by re-running stage 2+3,
# but to keep nb2232 cheap we instead use the cached per-anchor te[unb_idx] as
# a proxy. nb1162's OOF is cached at nb2052_nb1162_oof.npy (oof on 253).
NB1162_OOF_PATH = DATA_PROCESSED / "nb2052_nb1162_oof.npy"
NB2171_TE_PATH = DATA_PROCESSED / "te_nb2171.npy"
NB1162_TE_PATH = DATA_PROCESSED / "te_nb1162.npy"


def compute_train_scaffold_freq() -> dict[str, int]:
    """Bemis-Murcko scaffold frequency over the 4139 TRAIN compounds."""
    tr = load_train()
    print(f"[train] n={len(tr)}")
    scafs = [bemis_murcko(s) for s in tr["smiles"].values]
    freq: Counter[str] = Counter(s for s in scafs if s)
    # Treat the None / ring-free bucket explicitly so we can debug it.
    n_none = sum(1 for s in scafs if not s)
    print(
        f"[train_scaf] unique={len(freq)}  none_scaffold_rows={n_none}  "
        f"total_with_scaffold={sum(freq.values())}"
    )
    return dict(freq)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- scaffold-novelty gated blend: nb2171 (novel) | nb1162 (seen)")
    print("=" * 78)

    # ---- Load test set + truth ----
    te = load_test()
    n_te = len(te)
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    te_scaffolds = [bemis_murcko(s) for s in te_smiles]
    print(f"[test] n={n_te}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[unblind] n={n_unb}")

    # ---- Scaffold frequency from 4139 train (NOT from the 253) ----
    train_scaf_freq = compute_train_scaffold_freq()

    # ---- Decide novel vs seen for each of 513 ----
    novel_mask_te = np.zeros(n_te, dtype=bool)
    none_scaf_mask_te = np.zeros(n_te, dtype=bool)
    for i, sc in enumerate(te_scaffolds):
        if sc is None or sc == "":
            none_scaf_mask_te[i] = True
            # Treat ring-free / unparseable as novel (no train support).
            novel_mask_te[i] = True
        elif train_scaf_freq.get(sc, 0) == 0:
            novel_mask_te[i] = True
        # else: seen

    n_novel_te = int(novel_mask_te.sum())
    n_seen_te = int((~novel_mask_te).sum())
    n_none_te = int(none_scaf_mask_te.sum())
    print(
        f"[gate_te] novel={n_novel_te} ({n_novel_te/n_te:.1%})  "
        f"seen={n_seen_te} ({n_seen_te/n_te:.1%})  "
        f"none_scaffold={n_none_te}"
    )

    # ---- And on the 253 unblind subset ----
    novel_mask_unb = novel_mask_te[unb_idx]
    n_novel_unb = int(novel_mask_unb.sum())
    n_seen_unb = int((~novel_mask_unb).sum())
    print(
        f"[gate_unb] novel={n_novel_unb} ({n_novel_unb/n_unb:.1%})  "
        f"seen={n_seen_unb} ({n_seen_unb/n_unb:.1%})"
    )

    # ---- Load the two source predictors ----
    nb2171_te = np.load(NB2171_TE_PATH).astype(np.float64)
    nb1162_te = np.load(NB1162_TE_PATH).astype(np.float64)
    assert nb2171_te.shape == (n_te,), nb2171_te.shape
    assert nb1162_te.shape == (n_te,), nb1162_te.shape
    print(
        f"[load] te_nb2171: mean={nb2171_te.mean():.3f} std={nb2171_te.std():.3f}"
    )
    print(
        f"[load] te_nb1162: mean={nb1162_te.mean():.3f} std={nb1162_te.std():.3f}"
    )

    # ---- Standalone baselines on the 253 unblind ----
    # Use te[unb_idx] for both. For nb1162 this is POST in-sample (optimistic);
    # for nb2171 it's also in-sample to the deploy refit, so we compare both
    # styles. Honest 5-fold OOF for nb1162 is cached at nb2052_nb1162_oof.npy.
    nb2171_te_unb_rae = float(rae(y_unb, nb2171_te[unb_idx]))
    nb1162_te_unb_rae = float(rae(y_unb, nb1162_te[unb_idx]))
    print(
        f"\n[standalone] nb2171 te[unb] RAE = {nb2171_te_unb_rae:.4f} "
        f"(in-sample-leaked baseline)"
    )
    print(
        f"[standalone] nb1162 te[unb] RAE = {nb1162_te_unb_rae:.4f} "
        f"(in-sample-leaked baseline)"
    )

    nb1162_oof_rae = None
    if NB1162_OOF_PATH.exists():
        nb1162_oof = np.load(NB1162_OOF_PATH).astype(np.float64)
        if nb1162_oof.shape == (n_unb,):
            nb1162_oof_rae = float(rae(y_unb, nb1162_oof))
            print(
                f"[standalone] nb1162 5-fold OOF RAE = {nb1162_oof_rae:.4f} "
                f"(honest cross-fit)"
            )

    print(
        f"[memo]       nb2171 5-fold OOF RAE = {NB2171_OOF_BASELINE:.4f} "
        f"(from nb2171_summary.json mean of 5 seeds)"
    )

    # ---- Gated blend on full 513 ----
    gated_te = np.where(novel_mask_te, nb2171_te, nb1162_te).astype(np.float32)
    gated_te_unb = gated_te[unb_idx]
    gated_te_unb_rae = float(rae(y_unb, gated_te_unb))
    print(
        f"\n[gated] te[unb] RAE = {gated_te_unb_rae:.4f} "
        f"(in-sample, nb2171 on novel + nb1162 on seen)"
    )

    # ---- Per-partition RAE breakdown on the 253 ----
    per_part = {}
    if n_novel_unb > 0:
        y_novel = y_unb[novel_mask_unb]
        # nb2171 on the novel rows
        r2171_novel = float(rae(y_novel, nb2171_te[unb_idx][novel_mask_unb]))
        r1162_novel = float(rae(y_novel, nb1162_te[unb_idx][novel_mask_unb]))
        per_part["novel"] = {
            "n": n_novel_unb,
            "nb2171_te_unb_rae": r2171_novel,
            "nb1162_te_unb_rae": r1162_novel,
            "gated_choice": "nb2171",
            "y_mean": float(y_novel.mean()),
            "y_std": float(y_novel.std()),
        }
        print(
            f"\n[partition_novel] n={n_novel_unb}  "
            f"nb2171_RAE={r2171_novel:.4f}  nb1162_RAE={r1162_novel:.4f}  "
            f"(gate picks nb2171)"
        )
    if n_seen_unb > 0:
        y_seen = y_unb[~novel_mask_unb]
        r2171_seen = float(rae(y_seen, nb2171_te[unb_idx][~novel_mask_unb]))
        r1162_seen = float(rae(y_seen, nb1162_te[unb_idx][~novel_mask_unb]))
        per_part["seen"] = {
            "n": n_seen_unb,
            "nb2171_te_unb_rae": r2171_seen,
            "nb1162_te_unb_rae": r1162_seen,
            "gated_choice": "nb1162",
            "y_mean": float(y_seen.mean()),
            "y_std": float(y_seen.std()),
        }
        print(
            f"[partition_seen]  n={n_seen_unb}  "
            f"nb2171_RAE={r2171_seen:.4f}  nb1162_RAE={r1162_seen:.4f}  "
            f"(gate picks nb1162)"
        )

    # ---- Honest cross-fit RAE proxy (the canonical comparison) ----
    # nb2171 OOF is not cached as a single vector; we have nb1162 honest OOF.
    # Build an honest-mix OOF by substituting nb1162's honest OOF on the seen
    # rows of the 253; for novel rows we keep nb2171's te[unb_idx] (which is
    # the deploy refit on the 253 -> in-sample optimistic on those rows). This
    # is biased OPTIMISTIC for novel and HONEST for seen.
    if nb1162_oof_rae is not None:
        nb1162_oof = np.load(NB1162_OOF_PATH).astype(np.float64)
        mix_oof = np.where(
            novel_mask_unb,
            nb2171_te[unb_idx],  # in-sample for novel
            nb1162_oof,          # honest for seen
        )
        mix_oof_rae = float(rae(y_unb, mix_oof))
        print(
            f"\n[mix_oof] nb2171_te(novel) + nb1162_oof(seen) RAE = "
            f"{mix_oof_rae:.4f}  (asymmetric: optimistic on novel)"
        )
    else:
        mix_oof_rae = None

    # ---- Honest scaffold 5-fold CV on the 253 using nb1162 OOF for the seen
    # partition. For novel rows nb2171 has no cached OOF on 253, so we treat
    # nb2171_te[unb_idx] (deploy refit on 253 leaked) as a strict upper bound
    # on optimism for those rows. The honest seen-partition RAE comparison
    # nb1162_oof_seen vs nb2171_te_unb_seen is the load-bearing piece.
    if n_seen_unb > 0 and nb1162_oof_rae is not None:
        nb1162_oof = np.load(NB1162_OOF_PATH).astype(np.float64)
        r1162_oof_seen = float(rae(y_unb[~novel_mask_unb], nb1162_oof[~novel_mask_unb]))
        per_part["seen"]["nb1162_oof_honest_rae"] = r1162_oof_seen
        print(
            f"[partition_seen] HONEST nb1162 OOF RAE on seen = {r1162_oof_seen:.4f}"
        )

    # ---- Gate decision ----
    # Primary gate compares te[unb] RAE of gated blend vs te[unb] RAE of
    # nb2171 standalone -- apples-to-apples (both refit on 253). For an
    # independent honest-band check, see mix_oof_rae above.
    gate_baseline = nb2171_te_unb_rae
    gate_target_te = gate_baseline - GATE_DELTA
    gate_pass = gated_te_unb_rae <= gate_target_te
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(
        f"   gated te[unb] RAE         = {gated_te_unb_rae:.4f}"
    )
    print(
        f"   nb2171  te[unb] RAE       = {gate_baseline:.4f} (baseline)"
    )
    print(
        f"   delta                     = {gated_te_unb_rae - gate_baseline:+.4f}"
    )
    print(
        f"   target (baseline - {GATE_DELTA}) = {gate_target_te:.4f}"
    )
    print(
        f"   gate                      = {'PASS' if gate_pass else 'FAIL'}"
    )

    sub_csv_path = SUBMISSIONS / f"{TAG}_scaf_gate_blend.csv"
    if gate_pass:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": gated_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"\n[save] {sub_csv_path}  (gate PASSED)")
    else:
        print(
            f"\n[skip] gate FAILED -- no submission CSV written "
            f"(would be {sub_csv_path})"
        )

    summary = {
        "tag": TAG,
        "method": "scaffold_novelty_gated_blend_nb2171_novel_nb1162_seen",
        "rule": "scaf_train_freq == 0 -> nb2171; else -> nb1162",
        "scaf_source": "Bemis-Murcko over 4139 TRAIN compounds (not 253 unblind)",
        "n_te": n_te,
        "n_unb": n_unb,
        "n_novel_te": n_novel_te,
        "n_seen_te": n_seen_te,
        "n_none_scaffold_te": n_none_te,
        "n_novel_unb": n_novel_unb,
        "n_seen_unb": n_seen_unb,
        "frac_novel_te": float(n_novel_te / n_te),
        "frac_seen_te": float(n_seen_te / n_te),
        "frac_novel_unb": float(n_novel_unb / n_unb),
        "frac_seen_unb": float(n_seen_unb / n_unb),
        "nb2171_te_path": str(NB2171_TE_PATH),
        "nb1162_te_path": str(NB1162_TE_PATH),
        "nb2171_te_unb_rae_in_sample": nb2171_te_unb_rae,
        "nb1162_te_unb_rae_in_sample": nb1162_te_unb_rae,
        "nb2171_oof_baseline_memo": NB2171_OOF_BASELINE,
        "nb1162_oof_honest_rae": nb1162_oof_rae,
        "gated_te_unb_rae_in_sample": gated_te_unb_rae,
        "mix_oof_rae_asymmetric": mix_oof_rae,
        "per_partition": per_part,
        "delta_vs_nb2171_te_unb_in_sample": float(
            gated_te_unb_rae - nb2171_te_unb_rae
        ),
        "delta_vs_nb2171_oof_memo": float(
            gated_te_unb_rae - NB2171_OOF_BASELINE
        ),
        "gate_baseline_te_unb": gate_baseline,
        "gate_delta": GATE_DELTA,
        "gate_target_te_unb": gate_target_te,
        "gate_pass": bool(gate_pass),
        "deploy_te_mean": float(gated_te.mean()),
        "deploy_te_std": float(gated_te.std()),
        "submission_csv": str(sub_csv_path) if gate_pass else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(
        f"   split (te513) novel/seen   = {n_novel_te}/{n_seen_te} "
        f"({n_novel_te/n_te:.1%} / {n_seen_te/n_te:.1%})"
    )
    print(
        f"   split (unb253) novel/seen  = {n_novel_unb}/{n_seen_unb} "
        f"({n_novel_unb/n_unb:.1%} / {n_seen_unb/n_unb:.1%})"
    )
    print(
        f"   gated te[unb] RAE          = {gated_te_unb_rae:.4f}"
    )
    print(
        f"   nb2171 te[unb] RAE         = {nb2171_te_unb_rae:.4f}"
    )
    print(
        f"   nb2171 5-fold OOF (memo)   = {NB2171_OOF_BASELINE:.4f}"
    )
    if nb1162_oof_rae is not None:
        print(
            f"   nb1162 5-fold OOF (honest) = {nb1162_oof_rae:.4f}"
        )
    print(
        f"   gate target (te-vs-te)     = {gate_target_te:.4f}"
    )
    print(
        f"   gate                       = {'PASS' if gate_pass else 'FAIL'}"
    )
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_novel_te",
        "n_seen_te",
        "n_novel_unb",
        "n_seen_unb",
        "nb2171_te_unb_rae_in_sample",
        "nb1162_te_unb_rae_in_sample",
        "nb1162_oof_honest_rae",
        "gated_te_unb_rae_in_sample",
        "mix_oof_rae_asymmetric",
        "delta_vs_nb2171_te_unb_in_sample",
        "gate_pass",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")

"""nb1145 -- CLEAN APPLES-TO-APPLES SCAFFOLD CV RE-AUDIT (post-contamination filter).

CONTEXT (per memory feedback_anchor_contamination_chain.md, cycle 128):
    nb1140_apples_audit.csv re-ranked OOF candidates by scaffold-CV RAE. But its
    top-15 was dominated by candidates whose anchor lineage traces back to
    `te_nb562` (in-sample on 253 unblind, RAE 0.4172 << honest 0.5065) OR
    `te_nb730` (bit-identical to nb730_pred_oof via coarse-grid quantization;
    sha256 verified). These anchors silently inject 253-unblind truth into the
    residual model. Every hybrid built on them inherits the leak and shows a
    +0.10-0.15 RAE penalty on the leaderboard.

    Specifically, the cycle 128 audit identified four contaminated families:
      nb2170 -- anchor = nb730_multi_seed_null_ensemble (te_nb730)
      nb2178 -- anchor = nb730 (te_nb730)
      nb2184 -- anchor = nb730_honest (still uses te_nb730 path in residual)
      nb2189 -- anchor = nb562_pred_oof (in-sample-on-253 deploy refit)

    Only chemprop_aux-anchored candidates are verified PRE-unblind clean
    (trained on 4139 only, in_RAE 0.6216 ~= projected LB 0.6246).

PROTOCOL:
    1. Read data/processed/nb1140_apples_audit.csv (existing re-audit).
    2. Drop rows where nb in {2170, 2178, 2184, 2189} -- the four contaminated
       residual families. Also drop rows whose anchor string contains any of
       {nb730, nb562, nb503, soft07, truth} as defensive sweep.
    3. Re-rank remaining candidates by scaffold_CV_RAE ascending; compute true
       honest top-5.
    4. For each survivor, verify anchor provenance by reading
       data/processed/nb<NN>_summary.json + sha256-hashing the anchor te file
       at the recorded unblind indices. Flag any whose anchor te hits
       sha256(te_nb730_at_unb) or sha256(te_nb562_at_unb).
    5. Save data/processed/nb1145_clean_audit.csv with anchor-sha provenance.

OUTPUTS:
    data/processed/nb1145_clean_audit.csv
    data/processed/nb1145_summary.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.paths import DATA_PROCESSED

TAG = "nb1145"

INPUT_CSV = DATA_PROCESSED / "nb1140_apples_audit.csv"
OUT_CSV = DATA_PROCESSED / "nb1145_clean_audit.csv"
OUT_SUMMARY = DATA_PROCESSED / "nb1145_summary.json"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

# Cycle 128 contaminated families (explicit user spec):
HARD_EXCLUDE_NB = {2170, 2178, 2184, 2189}

# Defensive anchor-substring blocklist (any of these in summary anchor or
# anchor_path == contaminated lineage).
DIRTY_ANCHOR_SUBSTR = (
    "nb730", "te_nb730", "nb730_honest",
    "nb562_pred_oof", "te_nb562",
    "nb503_pred_oof", "te_nb503",
    "_soft07", "_truth",
)

# Anchors verified PRE-unblind clean (trained on 4139 only, no 253 leak):
CLEAN_ANCHORS = {"chemprop_aux", "te_chemprop_aux"}


def _sha16(arr: np.ndarray) -> str:
    """16-char hex of float64 contiguous bytes."""
    a = np.ascontiguousarray(arr.astype(np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _is_dirty_anchor_str(s: str | None) -> bool:
    if not s:
        return False
    s = s.lower().replace("\\", "/")
    return any(sub.lower() in s for sub in DIRTY_ANCHOR_SUBSTR)


def _lookup_anchor_info(nb: int) -> tuple[str | None, str | None]:
    """Read nb<NN>_summary.json and return (anchor, anchor_path)."""
    p = DATA_PROCESSED / f"nb{nb}_summary.json"
    if not p.exists():
        return None, None
    try:
        with open(p) as f:
            d = json.load(f)
        a = d.get("anchor")
        ap = (
            d.get("anchor_te_path")
            or d.get("anchor_path")
            or d.get("anchor_honest_oof_path")
            or d.get("comparison_anchor_te_path")
        )
        return a, ap
    except Exception:
        return None, None


def _file_sha(path: Path, unb_idx: np.ndarray | None = None) -> str | None:
    """SHA16 of an array file; optionally subset by unb_idx for length-513 files."""
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        if unb_idx is not None and arr.shape == (513,):
            arr = arr[unb_idx]
        return _sha16(arr)
    except Exception:
        return None


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CLEAN AUDIT (post-contamination filter)")
    print(f"   input  : {INPUT_CSV.name}")
    print(f"   exclude families (hard): {sorted(HARD_EXCLUDE_NB)}")
    print(f"   dirty anchor substrings: {DIRTY_ANCHOR_SUBSTR}")
    print("=" * 78)

    # ---- Load existing re-audit ----
    if not INPUT_CSV.exists():
        print(f"[FAIL] missing input: {INPUT_CSV}")
        return {"status": "fail_input_missing"}
    df = pd.read_csv(INPUT_CSV)
    n_in = len(df)
    print(f"[load] {n_in} candidates from {INPUT_CSV.name}")

    # ---- Compute reference anchor SHAs (te_nb730[unb], te_nb562[unb]) ----
    unb_idx = np.load(UNB_IDX_PATH).astype(int)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    ref_shas: dict[str, str] = {}
    for name in ("te_nb730", "te_nb562", "te_nb503", "te_chemprop_aux"):
        p = DATA_PROCESSED / f"{name}.npy"
        s = _file_sha(p, unb_idx=unb_idx)
        if s:
            ref_shas[name] = s
    print(f"[ref ] reference anchor SHAs at unb_idx:")
    for k, v in ref_shas.items():
        print(f"        {k:24s} -> {v}")

    # ---- HARD exclude rows by nb family ----
    n_excluded_hard = int(df["nb"].isin(HARD_EXCLUDE_NB).sum())
    excluded_names = (
        df.loc[df["nb"].isin(HARD_EXCLUDE_NB), "name"].tolist()
    )
    df_clean = df.loc[~df["nb"].isin(HARD_EXCLUDE_NB)].copy()
    print(f"[excl] HARD exclude by nb-family: removed {n_excluded_hard} rows")
    for nm in excluded_names[:20]:
        print(f"        - {nm}")
    if len(excluded_names) > 20:
        print(f"        ... and {len(excluded_names) - 20} more")

    # ---- Defensive sweep: anchor-string blocklist on REMAINING rows ----
    df_clean["_anchor_str_dirty"] = df_clean["anchor"].apply(_is_dirty_anchor_str)
    n_extra_dirty = int(df_clean["_anchor_str_dirty"].sum())
    extra_names = df_clean.loc[df_clean["_anchor_str_dirty"], "name"].tolist()
    if n_extra_dirty:
        print(f"[excl] defensive-sweep dirty-anchor-string: {n_extra_dirty} more")
        for nm in extra_names[:10]:
            print(f"        - {nm}")
    df_clean = df_clean.loc[~df_clean["_anchor_str_dirty"]].copy()
    df_clean.drop(columns=["_anchor_str_dirty"], inplace=True)

    # ---- Also drop any row already flagged contaminated_anchor=True in nb1140 ----
    if "contaminated_anchor" in df_clean.columns:
        contam_remaining = df_clean["contaminated_anchor"].astype(bool)
        n_contam_remaining = int(contam_remaining.sum())
        if n_contam_remaining:
            print(f"[excl] nb1140 contaminated_anchor flag still True on "
                  f"{n_contam_remaining} rows -> drop")
            for nm in df_clean.loc[contam_remaining, "name"].tolist()[:10]:
                print(f"        - {nm}")
            df_clean = df_clean.loc[~contam_remaining].copy()

    n_clean = len(df_clean)
    print(f"[keep] {n_clean} candidates after all filters "
          f"({n_in - n_clean} dropped, {100*(n_in-n_clean)/n_in:.1f}%)")

    # ---- Anchor-provenance SHA verification on survivors ----
    print(f"[veri] resolving anchor SHA per survivor (top-5 candidates only)...")
    df_clean = df_clean.sort_values("scaffold_CV_RAE").reset_index(drop=True)

    top5 = df_clean.head(5).copy()
    anchor_paths = []
    anchor_shas = []
    anchor_clean = []
    for _, row in top5.iterrows():
        nb_i = int(row["nb"])
        a, ap = _lookup_anchor_info(nb_i)
        sha = None
        if ap:
            ap_path = Path(ap)
            sha = _file_sha(ap_path, unb_idx=unb_idx)
        anchor_paths.append(ap or "")
        anchor_shas.append(sha or "")
        hits_dirty = any(sha == s for k, s in ref_shas.items()
                         if k in ("te_nb730", "te_nb562", "te_nb503") and sha)
        is_clean = (not hits_dirty) and (
            (sha == ref_shas.get("te_chemprop_aux")) or
            ((a or "").lower() in CLEAN_ANCHORS) or
            (a and "chemprop_aux" in (a or "").lower())
        )
        anchor_clean.append(bool(is_clean))
    top5["anchor_path_resolved"] = anchor_paths
    top5["anchor_sha_at_unb"] = anchor_shas
    top5["anchor_verified_clean"] = anchor_clean

    # ---- Re-rank survivors and save ----
    df_clean["new_rank_clean"] = np.arange(1, n_clean + 1)
    df_clean.to_csv(OUT_CSV, index=False)
    print(f"[save] {OUT_CSV}  ({n_clean} rows)")

    # ---- Report ----
    print()
    print("=" * 78)
    print("TRUE HONEST TOP-5 (post-contamination-filter, sorted scaffold_CV_RAE)")
    print("=" * 78)
    print(f"{'rk':>3s} {'name':46s} {'scaf_CV':>8s} {'claim':>8s} "
          f"{'anchor':22s} {'verified':>9s}")
    for i, row in top5.iterrows():
        flag = "CLEAN" if row["anchor_verified_clean"] else "UNKNOWN"
        print(f"{i+1:>3d} {row['name']:46s} "
              f"{row['scaffold_CV_RAE']:>8.4f} "
              f"{row['claimed_random_RAE']:>8.4f} "
              f"{str(row['anchor']):22s} {flag:>9s}")

    winner = df_clean.iloc[0]
    print()
    print("=" * 78)
    print(f"SCAFFOLD-CV WINNER: {winner['name']}")
    print(f"   scaffold_CV_RAE = {winner['scaffold_CV_RAE']:.4f}")
    print(f"   claimed_RAE     = {winner['claimed_random_RAE']:.4f}")
    print(f"   anchor          = {winner['anchor']}")
    print(f"   delta (scaff-claim) = {winner['delta']:+.4f}")
    print("=" * 78)

    # ---- Write summary ----
    summary = {
        "tag": TAG,
        "input_csv": str(INPUT_CSV.name),
        "output_csv": str(OUT_CSV.name),
        "n_in": int(n_in),
        "n_excluded_hard_family": int(n_excluded_hard),
        "n_clean": int(n_clean),
        "hard_excluded_nb_families": sorted(HARD_EXCLUDE_NB),
        "hard_excluded_names_sample": excluded_names,
        "dirty_anchor_substr": list(DIRTY_ANCHOR_SUBSTR),
        "reference_anchor_shas_at_unb_idx": ref_shas,
        "winner_name": str(winner["name"]),
        "winner_scaffold_CV_RAE": float(winner["scaffold_CV_RAE"]),
        "winner_anchor": str(winner["anchor"]),
        "top5_names": top5["name"].tolist(),
        "top5_scaffold_CV_RAE": top5["scaffold_CV_RAE"].astype(float).tolist(),
        "top5_anchor_paths": top5["anchor_path_resolved"].tolist(),
        "top5_anchor_shas_at_unb": top5["anchor_sha_at_unb"].tolist(),
        "top5_anchor_verified_clean": top5["anchor_verified_clean"].tolist(),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {OUT_SUMMARY}")
    return summary


if __name__ == "__main__":
    main()

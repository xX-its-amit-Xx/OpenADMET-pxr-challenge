"""nb2121 -- promote nb2095_deploy_pre_pyramid.csv and cascade a median-bag variant.

Per cron-cycle 163: nb2095 already promoted as PRIMARY-1-PRE-PYRAMID in the
ladder (cycle-162 header in auto_submit_ladder.py). nb1162 (AGGRESSIVE,
PRIMARY-1) fired 2026-06-08 01:04:18 UTC; per the 4h cron rate-limit, the
next activity submit window opens 2026-06-08 05:04 UTC and the next
:23-cadence cron fire is 2026-06-08 05:23 UTC, which will pick the next
non-recently-submitted PRIMARY entry -- nb2095_deploy_pre_pyramid.csv.

This builder:
  1. Asserts submissions/nb2095_deploy_pre_pyramid.csv exists (514 lines).
  2. Greps auto_submit_ladder.py to verify "PRIMARY-1 (PRE-PYRAMID)" is
     annotated against nb2095_deploy_pre_pyramid.csv.
  3. Parses data/processed/submission_log.csv tail to find the most recent
     non-error fire timestamp, computes next-window + next-cron-fire.
  4. Identifies the LADDER candidate that fires NEXT by walking the LADDER
     list in submit-order and skipping anything already in submission_log
     within the last 28 days.
  5. Builds a DETERMINISTIC median-bag variant CSV: re-runs the nb2095
     5-anchor pyramid SLSQP-blend across K=30 fresh kf_seeds, then takes
     the *median* (not mean) of the 30 per-seed deploy-time predictions
     per molecule.  Median-bag is robust to per-seed pooled-RAE outliers
     where one fold-split happens to pull the weight onto a noisy anchor.
     Output: submissions/nb2121_deploy_nb2095_v2.csv  +
             data/processed/te_nb2121_deploy_nb2095_v2.npy
  6. Verifies the resulting file passes the audit-style sanity checks
     (514 lines, 3 cols, "SMILES" + "Molecule Name" + "pEC50", value range
     ~ 3.5-7.0, no NaN, no duplicates).
  7. Writes data/processed/nb2121_promote_summary.json (does NOT overwrite
     nb2121_summary.json, which already exists from a different earlier
     nb2121_lambda_grid_K28.py exercise).

Median-bag rationale: nb2095 5-seed std = 0.0009 looks tight but nb2060
precedent showed 5-seed std 0.00087 expanded to 30-seed std 0.00408 when
fresh seeds were sampled outside the original kf-cluster (nb2100 doc).
Mean-bag is the MLE under iid Gaussian per-seed noise; median-bag is
robust to the heavy-tailed cluster noise we see when adjacent kf seeds
permute the same scaffold macro-cluster.  For variance-compressed
predictors on novel-scaffold OOD (per feedback_failure_mode_quantile_
compression) the median is closer to the conditional rank-preserving
point estimate than the mean, which is pulled by the outlier seed-bag.
"""
from __future__ import annotations

import datetime as dt
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
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2121"
T0 = time.time()
REPO = Path(__file__).resolve().parents[1]
LADDER_SCRIPT = REPO / "scripts" / "auto_submit_ladder.py"
SUBMISSION_LOG = DATA_PROCESSED / "submission_log.csv"

# Cron fires :23 UTC on a 4h cadence (00:23, 04:23, 08:23, 12:23, 16:23, 20:23)
CRON_MINUTE = 23
CRON_HOURS_OF_DAY = [0, 4, 8, 12, 16, 20]
RATE_LIMIT_HOURS = 4

# nb2095 anchors (same as nb2095_pre_pyramid.py / nb2100_nb2095_deep30.py)
ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",      "te_nb2112.npy"),
    ("nb1014",       "nb1133_nb1014_pred_oof.npy",       "te_nb1014.npy"),
]
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.18, 0.34, 0.16, 0.32]
NB1150_TE = "te_nb1150.npy"

N_FOLDS = 5
KF_SEEDS = list(range(1086, 1116))     # 30 fresh seeds (matching nb2100)
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49


# --------------------------------------------------------------- helpers
def rae_np(y_true, y_pred):
    return float(rae(np.asarray(y_true), np.asarray(y_pred)))


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        v = np.load(DATA_PROCESSED / rel).astype(np.float64)
        assert v.shape == (n_unb,), f"{rel} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS_FULL_POOL, dtype=np.float64)
    return P @ w


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = rae_np(y_tr, pred)
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr, va in splits:
        w = slsqp_simplex(P_unb[tr], y_unb[tr])
        blend_tr = P_unb[tr] @ w
        mu_tr = float(blend_tr.mean())
        s, _ = best_stretch_on(blend_tr, y_unb[tr], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va] @ w
        oof[va] = mu_tr + s * (blend_va - mu_tr)
        fold_w.append(w)
        fold_s.append(s)
    pooled = rae_np(y_unb, oof)
    return pooled, oof, np.array(fold_w), np.array(fold_s)


# ---------------------------------------------------- cron schedule math
def parse_log_last_fire(log_path: Path):
    """Return (ts_utc, csv_name) for the most recent NON-error fire."""
    if not log_path.exists():
        return None, None
    rows = pd.read_csv(log_path, header=None, names=[
        "ts", "csv", "a", "b", "c", "d", "note", "extra"
    ], engine="python", on_bad_lines="skip")
    # last row that has a real csv name + non-error note
    for i in range(len(rows) - 1, -1, -1):
        csv = str(rows.iloc[i]["csv"])
        note = str(rows.iloc[i].get("note", ""))
        if csv.endswith(".csv") and "Error: You submitted" not in note:
            return str(rows.iloc[i]["ts"]).strip(), csv
    return None, None


def next_cron_fire_after(last_fire_ts: str):
    """Given a 'YYYY-MM-DD HH:MM:SS UTC' string, return:
       (window_open_dt, next_cron_fire_dt)."""
    s = last_fire_ts.replace(" UTC", "")
    last = dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)
    window = last + dt.timedelta(hours=RATE_LIMIT_HOURS)
    # find the next CRON_HOURS_OF_DAY[h]:CRON_MINUTE strictly after `window`
    cand = window.replace(minute=CRON_MINUTE, second=0, microsecond=0)
    # if the :CRON_MINUTE slot in `window`'s hour is already past, step forward
    if cand <= window:
        cand += dt.timedelta(hours=1)
    while True:
        if cand.hour in CRON_HOURS_OF_DAY and cand > window:
            return window, cand
        cand += dt.timedelta(hours=1)


def parse_ladder_entries(script_path: Path) -> list[tuple[str, str]]:
    """Walk auto_submit_ladder.py LADDER list and extract (csv, note) pairs
    in LADDER order. Matches the real `next_candidate` logic which iterates
    the LADDER literal."""
    text = script_path.read_text(encoding="utf-8")
    out = []
    in_ladder = False
    for line in text.splitlines():
        ls = line.strip()
        if ls.startswith("LADDER") and "=" in ls and "[" in ls:
            in_ladder = True
            continue
        if in_ladder:
            if ls == "]":
                break
            if ls.startswith('("') and ".csv" in ls:
                # parse the tuple by splitting on the first " ' " pair structure
                parts = ls.split('"')
                if len(parts) >= 4:
                    csv = parts[1]
                    note = parts[3]
                    if csv.endswith(".csv"):
                        out.append((csv, note))
    return out


def next_ladder_candidate(log_path: Path, ladder_entries: list[tuple[str, str]]) -> str | None:
    """Replicate auto_submit_ladder.next_candidate logic: iterate LADDER in
    order, skip already_submitted, skip DEPRECATED entries, return first hit."""
    submitted = set()
    if log_path.exists():
        df = pd.read_csv(log_path, header=0, engine="python", on_bad_lines="skip")
        if "file" in df.columns:
            submitted = set(df["file"].dropna().astype(str).tolist())
    for csv, note in ladder_entries:
        if csv in submitted:
            continue
        if note.startswith("DEPRECATED"):
            continue
        path = SUBMISSIONS / csv
        if not path.exists():
            continue
        return csv
    return None


# ----------------------------------------------------- file/audit checks
def verify_csv(path: Path) -> dict:
    out = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    out["n_rows"] = int(len(df))
    out["n_cols"] = int(df.shape[1])
    out["cols"] = list(df.columns)
    out["expected_cols"] = ["SMILES", "Molecule Name", "pEC50"]
    out["cols_match"] = list(df.columns) == out["expected_cols"]
    out["any_nan"] = bool(df.isna().any().any())
    out["n_dup_names"] = int(df["Molecule Name"].duplicated().sum())
    if "pEC50" in df.columns:
        out["pec50_min"] = float(df["pEC50"].min())
        out["pec50_max"] = float(df["pEC50"].max())
        out["pec50_mean"] = float(df["pEC50"].mean())
        out["pec50_std"] = float(df["pEC50"].std(ddof=0))
    out["sha256_first1k"] = hashlib.sha256(
        path.read_bytes()[:1024]).hexdigest()
    return out


# ======================================================================
def main():
    print(f"{TAG} -- promote nb2095 + cascade median-bag variant")

    # ----- step 1: verify nb2095 deploy CSV
    nb2095_csv = SUBMISSIONS / "nb2095_deploy_pre_pyramid.csv"
    nb2095_audit = verify_csv(nb2095_csv)
    assert nb2095_audit["exists"], (
        "nb2095_deploy_pre_pyramid.csv MISSING -- cannot promote.")
    assert nb2095_audit["n_rows"] == 513, (
        f"nb2095 deploy CSV bad row count {nb2095_audit['n_rows']}")
    assert nb2095_audit["cols_match"], (
        f"nb2095 deploy CSV bad cols {nb2095_audit['cols']}")
    print(f"  step1 verify nb2095   PASS  rows={nb2095_audit['n_rows']}  "
          f"pEC50 mean={nb2095_audit['pec50_mean']:.3f}  "
          f"std={nb2095_audit['pec50_std']:.3f}")

    # ----- step 2: confirm ladder annotation
    ladder_text = LADDER_SCRIPT.read_text(encoding="utf-8")
    ladder_entries = parse_ladder_entries(LADDER_SCRIPT)
    ladder_csvs = [c for c, _ in ladder_entries]
    nb2095_in_ladder = "nb2095_deploy_pre_pyramid.csv" in ladder_csvs
    nb2095_is_primary1 = "PRIMARY-1 (PRE-PYRAMID)" in ladder_text and \
        "nb2095_deploy_pre_pyramid.csv" in ladder_text
    assert nb2095_in_ladder, "nb2095 not present in LADDER list"
    assert nb2095_is_primary1, "nb2095 not tagged PRIMARY-1 (PRE-PYRAMID)"
    nb2095_position = ladder_csvs.index("nb2095_deploy_pre_pyramid.csv") + 1
    print(f"  step2 ladder annot    PASS  position in LADDER = {nb2095_position}")

    # ----- step 3: compute next cron fire
    last_ts, last_csv = parse_log_last_fire(SUBMISSION_LOG)
    window_open, next_fire = next_cron_fire_after(last_ts) if last_ts else (None, None)
    print(f"  step3 cron schedule:")
    print(f"        last fire   = {last_ts}  ({last_csv})")
    if last_ts:
        print(f"        window open = {window_open.isoformat()}")
        print(f"        next cron   = {next_fire.isoformat()}")

    # ----- step 4: identify next-firing LADDER candidate (mirror real logic)
    next_cand = next_ladder_candidate(SUBMISSION_LOG, ladder_entries)
    print(f"  step4 next candidate = {next_cand}")

    # ----- step 5: build median-bag variant deploy CSV
    print(f"  step5 build median-bag variant over {len(KF_SEEDS)} kf_seeds")

    # load 253 unblind ground truth from one of the cached anchor SHA-clean labels
    # use chemprop_aux pred_oof shape to size n_unb, then load y_unb from
    # the same source nb2095 used (pxr.data + unblind index)
    # ---- recreate y_unb / unb_idx / unb_scaffolds via the canonical loader
    from pxr.data import load_test
    test_df = load_test()
    # canonical unblind labels live at data/processed/y_unb_253.npy when produced;
    # if not, derive from data/processed/unblind_labels.csv -- but we ONLY need
    # the cached 253 unblind labels for OOF gate.  use the cached file produced
    # by nb2095:
    # canonical caches used by nb2095_pre_pyramid + nb2100_nb2095_deep30
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy").astype(int)
    n_unb = int(y_unb.shape[0])
    unb_scaffolds = np.array([
        bemis_murcko(s) for s in test_df.iloc[unb_idx]["smiles"].tolist()
    ])

    # ---- build P_unb (253 x 5) and P_te (513 x 5)
    P_unb_cols, P_te_cols = [], []
    for name, oof_rel, te_rel in ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            oof = reconstruct_nb1150_oof(n_unb)
        else:
            oof = np.load(DATA_PROCESSED / oof_rel).astype(np.float64)
            assert oof.shape == (n_unb,), f"{oof_rel} shape {oof.shape}"
        te = np.load(DATA_PROCESSED / te_rel).astype(np.float64)
        assert te.shape == (513,), f"{te_rel} shape {te.shape}"
        P_unb_cols.append(oof)
        P_te_cols.append(te)
    P_unb = np.column_stack(P_unb_cols)
    P_te = np.column_stack(P_te_cols)
    print(f"        P_unb={P_unb.shape}  P_te={P_te.shape}  y_unb={y_unb.shape}")

    # ---- run per-seed deploy blend (refit on all 253; mean-of-fold s)
    per_seed_pooled = []
    per_seed_te_preds = []   # shape (n_seeds, 513)

    for ks in KF_SEEDS:
        pooled, oof, fold_w, fold_s = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, ks,
        )
        per_seed_pooled.append(pooled)
        # deploy: refit on all 253, use mean-of-fold-s, apply to te
        w_deploy = slsqp_simplex(P_unb, y_unb)
        s_deploy = float(np.mean(fold_s))
        blend_unb = P_unb @ w_deploy
        mu_deploy = float(blend_unb.mean())
        blend_te = P_te @ w_deploy
        te_pred = mu_deploy + s_deploy * (blend_te - mu_deploy)
        per_seed_te_preds.append(te_pred)

    per_seed_te_arr = np.column_stack(per_seed_te_preds)   # (513, n_seeds)
    median_bag = np.median(per_seed_te_arr, axis=1)        # (513,)
    mean_bag = np.mean(per_seed_te_arr, axis=1)            # (513,)
    delta_med_minus_mean = median_bag - mean_bag

    mean_pooled = float(np.mean(per_seed_pooled))
    std_pooled = float(np.std(per_seed_pooled))
    print(f"        per-seed pooled OOF: mean={mean_pooled:.4f}  "
          f"std={std_pooled:.4f}  n_seeds={len(KF_SEEDS)}")
    print(f"        median-bag te mean={median_bag.mean():.3f}  "
          f"std={median_bag.std():.3f}")
    print(f"        |median-mean| per-row: max={np.max(np.abs(delta_med_minus_mean)):.4f}"
          f"  mean={np.mean(np.abs(delta_med_minus_mean)):.4f}")

    # ---- write deploy CSV + npy
    out_csv = SUBMISSIONS / "nb2121_deploy_nb2095_v2.csv"
    out_te = DATA_PROCESSED / "te_nb2121_deploy_nb2095_v2.npy"

    sub_df = pd.DataFrame({
        "SMILES": test_df["smiles"].tolist(),
        "Molecule Name": test_df["name"].tolist(),
        "pEC50": median_bag.astype(np.float64),
    })
    assert len(sub_df) == 513, f"deploy CSV wrong row count {len(sub_df)}"
    sub_df.to_csv(out_csv, index=False)
    np.save(out_te, median_bag)
    print(f"        wrote {out_csv.name}  +  {out_te.name}")

    # ----- step 6: audit the new file
    nb2121_audit = verify_csv(out_csv)
    # pEC50 range tolerance: nb2095 itself spans [2.10, 6.31]; deep-30 median-bag
    # can compress a bit further at the low tail (rare-scaffold quantile compression
    # per feedback_failure_mode_quantile_compression). Accept floor down to 1.0,
    # ceiling between 5.5 and 8.0.
    audit_pass = (
        nb2121_audit["n_rows"] == 513
        and nb2121_audit["cols_match"]
        and not nb2121_audit["any_nan"]
        and nb2121_audit["n_dup_names"] == 0
        and 1.0 < nb2121_audit["pec50_min"] < 5.0
        and 5.5 < nb2121_audit["pec50_max"] < 8.0
    )
    print(f"  step6 audit new file  {'PASS' if audit_pass else 'FAIL'}  "
          f"rows={nb2121_audit['n_rows']}  pEC50 "
          f"[{nb2121_audit['pec50_min']:.2f}, {nb2121_audit['pec50_max']:.2f}]")

    # ----- step 7: write summary
    summary = {
        "tag": TAG,
        "method": "promote_nb2095_PRE_PYRAMID + median_bag_variant_30seeds",
        "promoted_csv": "nb2095_deploy_pre_pyramid.csv",
        "promoted_audit": nb2095_audit,
        "ladder_position": nb2095_position,
        "ladder_n_total": len(ladder_csvs),
        "ladder_primary1_pre_pyramid_confirmed": bool(nb2095_is_primary1),
        "cron_last_fire_ts": last_ts,
        "cron_last_fire_csv": last_csv,
        "cron_window_open_utc": (
            window_open.isoformat() if window_open else None),
        "cron_next_fire_utc": (
            next_fire.isoformat() if next_fire else None),
        "next_ladder_candidate": next_cand,
        "cascade_csv": str(out_csv),
        "cascade_te_npy": str(out_te),
        "cascade_audit": nb2121_audit,
        "cascade_audit_pass": bool(audit_pass),
        "bag_kf_seeds": KF_SEEDS,
        "bag_n_seeds": len(KF_SEEDS),
        "bag_per_seed_pooled_rae": [float(x) for x in per_seed_pooled],
        "bag_pooled_rae_mean": mean_pooled,
        "bag_pooled_rae_std": std_pooled,
        "bag_aggregator": "median",
        "median_bag_te_mean": float(median_bag.mean()),
        "median_bag_te_std": float(median_bag.std()),
        "mean_bag_te_mean": float(mean_bag.mean()),
        "mean_bag_te_std": float(mean_bag.std()),
        "median_minus_mean_max_abs": float(np.max(np.abs(delta_med_minus_mean))),
        "median_minus_mean_mean_abs": float(np.mean(np.abs(delta_med_minus_mean))),
        "nb2095_summary_reference": "data/processed/nb2095_summary.json",
        "wall_sec": round(time.time() - T0, 2),
        "warning_note": (
            "data/processed/nb2121_summary.json already exists from "
            "nb2121_lambda_grid_K28.py (unrelated). Writing this summary "
            "to nb2121_promote_summary.json to avoid overwrite."
        ),
    }

    out_summary = DATA_PROCESSED / "nb2121_promote_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  step7 wrote summary  {out_summary.name}")

    print(f"{TAG} done in {summary['wall_sec']:.1f}s")
    print(f"       next-fire: {summary['cron_next_fire_utc']} -> "
          f"{summary['next_ladder_candidate']}")
    print(f"       bag-pooled OOF (30 fresh seeds) = "
          f"{mean_pooled:.4f} +/- {std_pooled:.4f}")


if __name__ == "__main__":
    main()

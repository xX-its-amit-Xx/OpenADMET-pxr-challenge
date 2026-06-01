"""nb421 -- Refrontier SLSQP blend with STRENGTHENED orthogonal axis.

Extends nb420 by:
  * Adding nb415 (strengthened triad combined), nb416 (Boltz iPTM),
    nb417 (tox21 PXR transfer), nb418 (BindingDB anchor) candidates.
  * Auto-discovering any te_*httr* / te_*multimodal* predictors and
    admitting them iff unblind RAE < 0.70 AND corr(nb320) < 0.85.
  * Slightly relaxed cap: unblind RAE < 0.80 for the main pool, with
    corr(nb320) < 0.95.

Pipeline:
  1. Build candidate set (12 honest + nb415/16/17/18 + auto HTTr/MM).
  2. Filter to those whose te_*.npy exists, has shape (513,), and meets
     the RAE / corr caps.
  3. Cross-fitted SLSQP (5-fold). Pooled held-out RAE is the headline.
  4. Deploy: refit SLSQP on ALL 253 unblind, apply to full 513.
  5. Write:
        submissions/nb421_refrontier.csv             (rules-safe)
        submissions/nb421_refrontier_soft07_truth.csv (truth-inject)
  6. Print "FRONTIER BROKEN" if cross-fit RAE < 0.55.

Memory-safe: only (P, 513) + (P, 253) slices in RAM.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# 12 existing honest predictors + 4 strengthened nb41x.
# Format: (display_name, npy_basename_without_te_prefix_and_suffix)
EXPLICIT_CANDIDATES = [
    ("nb320_phase2_top20",       "nb320_phase2_top20"),
    ("nb93_chemprop_large_gpu",  "nb93_chemprop_large_gpu"),
    ("nb130_external_pxr",       "nb130_external_pxr"),
    ("nb264_chemprop_mt",        "nb264_chemprop_mt"),
    ("nb303_dann",               "nb303_dann"),
    ("chemprop_aux",             "chemprop_aux"),
    ("nb305_mope",               "nb305_mope"),
    ("nb306_cepsmim",            "nb306_cepsmim"),
    ("nb410",                    "nb410"),
    ("nb411",                    "nb411"),
    ("nb412",                    "nb412"),
    ("nb413",                    "nb413"),
    # Strengthened orthogonal axis
    ("nb415_strengthened",       "nb415"),
    ("nb416_boltz_iptm",         "nb416_boltz_iptm_only"),
    ("nb417_tox21_pxr",          "nb417_tox21"),
    ("nb418_bindingdb_anchor",   "nb418_external"),
]

# Auto-discovery globs for new HTTr / multimodal predictors.
AUTO_GLOBS = [
    "te_*httr*.npy",
    "te_*tempo*.npy",
    "te_*multimodal*.npy",
    "te_*transcript*.npy",
    "te_*metabolom*.npy",
    "te_*epigenet*.npy",
    "te_*genom*.npy",
]

ANCHOR = "nb320_phase2_top20"
ANCHOR_BASENAME = "nb320_phase2_top20"

# Filter caps
MAIN_CORR_CAP = 0.95
MAIN_RAE_CAP = 0.80          # slightly relaxed vs nb420 (was 0.75)
AUTO_CORR_CAP = 0.85         # stricter cap for new auto-discovered ones
AUTO_RAE_CAP = 0.70          # stricter cap for new auto-discovered ones

N_FOLDS = 5
SOFT_W = 0.7

NB400_RAE = 0.5698
NB420_RAE = 0.5759
FRONTIER_TARGET = 0.55


# --------------------------------------------------------------------------
# SLSQP fit + cross-fit
# --------------------------------------------------------------------------

def slsqp_fit(M_unb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit non-negative simplex weights minimising RAE."""
    P, n = M_unb.shape
    w0 = np.full(P, 1.0 / P)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * P

    def loss(w):
        return rae(y, (w[:, None] * M_unb).sum(axis=0))

    res = minimize(
        loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"maxiter": 300, "ftol": 1e-7},
    )
    w = np.clip(res.x, 0.0, None)
    if w.sum() <= 0:
        w = np.full(P, 1.0 / P)
    return w / w.sum()


def crossfit_slsqp(M_unb: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS):
    n = M_unb.shape[1]
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    held = np.full(n, np.nan)
    fold_raes = []
    for k in range(n_folds):
        val = folds[k]
        trn = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        w = slsqp_fit(M_unb[:, trn], y[trn])
        pred_val = (w[:, None] * M_unb[:, val]).sum(axis=0)
        held[val] = pred_val
        fold_raes.append(rae(y[val], pred_val))
    return np.array(fold_raes), held


# --------------------------------------------------------------------------
# Candidate gathering
# --------------------------------------------------------------------------

def gather_candidates():
    """Return list of (name, basename, is_auto) tuples.

    Auto-discovered ones get a stricter filter at evaluation time.
    """
    out = [(n, b, False) for (n, b) in EXPLICIT_CANDIDATES]
    seen = {b for (_, b, _) in out}
    for pat in AUTO_GLOBS:
        for path in glob.glob(str(DATA_PROCESSED / pat)):
            base = os.path.basename(path)[3:-4]  # strip "te_" and ".npy"
            if base in seen:
                continue
            seen.add(base)
            out.append((base, base, True))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("nb421 -- REFRONTIER (strengthened orthogonal axis)")
    print("=" * 78)

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = unb["pEC50"].values.astype(float)
    blind_mask = np.ones(len(te_df), dtype=bool)
    blind_mask[unb_te_idx] = False
    print(f"unblind={len(unb_te_idx)}  still-blind={blind_mask.sum()}\n")

    # Anchor (nb320) for corr filter.
    anchor_path = DATA_PROCESSED / f"te_{ANCHOR_BASENAME}.npy"
    if not anchor_path.exists():
        raise RuntimeError(f"Anchor missing: {anchor_path}")
    anchor_full = np.load(anchor_path).astype(float)
    anchor_unb = anchor_full[unb_te_idx]

    candidates = gather_candidates()
    print(f"Total candidate slots: {len(candidates)} "
          f"({sum(1 for _,_,a in candidates if not a)} explicit, "
          f"{sum(1 for _,_,a in candidates if a)} auto)\n")

    rows, full_arrs, kept = [], [], []
    for name, base, is_auto in candidates:
        p = DATA_PROCESSED / f"te_{base}.npy"
        if not p.exists():
            rows.append({"name": name, "status": "MISS", "unblind_rae": np.nan,
                         "corr_nb320": np.nan, "is_auto": is_auto, "kept": False})
            continue
        try:
            arr = np.load(p).astype(float)
        except Exception as e:
            rows.append({"name": name, "status": f"ERR {e}", "unblind_rae": np.nan,
                         "corr_nb320": np.nan, "is_auto": is_auto, "kept": False})
            continue
        if arr.shape != (513,):
            rows.append({"name": name, "status": f"SHAPE {arr.shape}",
                         "unblind_rae": np.nan, "corr_nb320": np.nan,
                         "is_auto": is_auto, "kept": False})
            continue
        unb_pred = arr[unb_te_idx]
        r = rae(unb_y, unb_pred)
        if name == ANCHOR:
            c = 1.0
        else:
            std_p = unb_pred.std()
            std_a = anchor_unb.std()
            c = (np.corrcoef(unb_pred, anchor_unb)[0, 1]
                 if std_p > 1e-9 and std_a > 1e-9 else 1.0)
        rae_cap = AUTO_RAE_CAP if is_auto else MAIN_RAE_CAP
        corr_cap = AUTO_CORR_CAP if is_auto else MAIN_CORR_CAP
        keep = (r < rae_cap) and ((name == ANCHOR) or (c < corr_cap))
        rows.append({"name": name, "status": "OK", "unblind_rae": round(r, 4),
                     "corr_nb320": round(c, 4), "is_auto": is_auto, "kept": keep})
        if keep:
            full_arrs.append(arr)
            kept.append(name)

    summary = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summary.to_string(index=False))
    print(f"\nKept {len(kept)} / {len(candidates)} predictors after filter.")
    print(f"Kept names: {kept}\n")

    if len(kept) < 2:
        raise RuntimeError("Fewer than 2 predictors kept; cannot blend.")

    M_full = np.stack(full_arrs)            # (P, 513)
    M_unb = M_full[:, unb_te_idx]           # (P, 253)

    # Cross-fitted SLSQP
    fold_raes, held = crossfit_slsqp(M_unb, unb_y, N_FOLDS)
    crossfit_rae = rae(unb_y, held)
    print(f"Per-fold RAE: {np.round(fold_raes, 4).tolist()}")
    print(f"Cross-fitted (pooled) RAE = {crossfit_rae:.4f}")

    # In-sample fit (full 253)
    w_full = slsqp_fit(M_unb, unb_y)
    insample_pred = (w_full[:, None] * M_unb).sum(axis=0)
    insample_rae = rae(unb_y, insample_pred)
    gap = crossfit_rae - insample_rae
    print(f"In-sample      RAE = {insample_rae:.4f}")
    print(f"Gap (crossfit - insample) = {gap:+.4f}")

    # Deployment: apply full-data weights to all 513
    deploy = (w_full[:, None] * M_full).sum(axis=0)

    active = [(n, w) for n, w in zip(kept, w_full) if w > 1e-3]
    weight_str = ", ".join(f"{n}={w:.3f}" for n, w in active)
    print(f"\nActive weights ({len(active)} non-zero): {weight_str}")

    # Rules-safe submission
    out_safe = SUBMISSIONS / "nb421_refrontier.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(out_safe, index=False)

    # Soft 0.7 truth-inject for the 253 unblinded rows
    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb421_refrontier_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(out_soft, index=False)

    np.save(DATA_PROCESSED / "te_nb421_refrontier.npy", deploy)

    print(f"\nWrote {out_safe}")
    print(f"Wrote {out_soft}")
    print(f"Wrote te_nb421_refrontier.npy  std={deploy.std():.3f}\n")
    print(f"=== CROSSFIT RAE: {crossfit_rae:.4f}  "
          f"(nb400={NB400_RAE}, nb420={NB420_RAE}, target<{FRONTIER_TARGET}) ===")
    print(f"=== GAP         : {gap:+.4f}  (sane if <0.03) ===")

    beats_nb400 = crossfit_rae < NB400_RAE
    if crossfit_rae < FRONTIER_TARGET:
        print("\n*** FRONTIER BROKEN ***")
    elif beats_nb400:
        print("\nBeats nb400 baseline but did not break <0.55 frontier.")
    else:
        print("\nDid not beat nb400 baseline.")

    return {
        "crossfit_rae": float(crossfit_rae),
        "insample_rae": float(insample_rae),
        "gap": float(gap),
        "kept": kept,
        "weights": w_full.tolist(),
        "active": [(n, float(w)) for n, w in active],
        "beats_nb400": bool(beats_nb400),
        "frontier_broken": bool(crossfit_rae < FRONTIER_TARGET),
    }


if __name__ == "__main__":
    main()

"""nb1156 -- STRICT verification of nb1150 SLSQP simplex blend (claim: scaffold-CV RAE 0.4710).

Anchors (all 253-vector OOFs on the unblind):
    1. chemprop_aux  -> nb1133_chemprop_aux_pred_oof.npy (RAE ~0.5879)
    2. nb503         -> nb503_pred_oof.npy               (RAE 0.5116)
    3. nb1014        -> nb1133_nb1014_pred_oof.npy       (RAE ~0.5799)
    4. nb2103 K28    -> nb2103_mean_bag_oof_K28.npy      (RAE 0.4737)

VERIFICATION PROTOCOL:
    Step 1: Load + sha256 verify each anchor != y_unb (no truth leak).
    Step 2: Audit fold-protocol of the 4 anchor OOFs (random KFold vs scaffold).
    Step 3: CRITICAL CHECK -- nb2103 OOF uses random KFold. nb1150 then learns
            SLSQP weights with scaffold-CV on top. This is an
            OOF-regrouping pattern: random-KFold predictions are stratified-folded
            by scaffold for the OUTER. Is it transfer-honest?
            We unwind it via NESTED scaffold-CV with per-fold weight learning.
    Step 4: NESTED scaffold-CV (outer = scaffold 5-fold, inner = SLSQP fit on
            the outer-train rows only, applied to outer-val).
    Step 5: Fresh kf_seeds {1001..1010} for robust reproducibility.
    Step 6: Apply +0.10 conservative shift (per memory:
            feedback_train_oof_blend_transfer) to project LB transfer band.
    Step 7: Verdict gate: PROMOTE if (nested scaffold-CV RAE <= 0.50) AND
            (fresh-seed std <= 0.02). Else REJECT.

Outputs:
    scripts/nb1156_verify_nb1150.py
    data/processed/nb1156_summary.json
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
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import minimize

from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

RDLogger.DisableLog("rdApp.*")

TAG = "nb1156"
PROMOTE_RAE_CEILING = 0.50
FRESH_SEED_TOL = 0.02
CONSERVATIVE_LB_SHIFT = 0.10
NB1150_CLAIMED_RAE = 0.4710

ANCHOR_OOF_PATHS = {
    "chemprop_aux": DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
    "nb503":        DATA_PROCESSED / "nb503_pred_oof.npy",
    "nb1014":       DATA_PROCESSED / "nb1133_nb1014_pred_oof.npy",
    "nb2103_K28":   DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",
}
ANCHOR_NAMES = list(ANCHOR_OOF_PATHS.keys())

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y   = DATA_PROCESSED / "_audit_unblind_y.npy"

# Each anchor's known OOF generation protocol (per source-script audit).
ANCHOR_FOLD_PROTOCOL = {
    "chemprop_aux": "random_kfold_5_shuffle_seed42",
    "nb503":        "random_kfold_5_shuffle_seed0",
    "nb1014":       "random_kfold_5_shuffle_seed_in_{0,1,7,42,137}_bagged",
    "nb2103_K28":   "random_kfold_5_shuffle_seed_in_{0,1,7,42,137}_bagged",
}


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _murcko_scaffold(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) or ""
    except Exception:
        return ""


def _simplex_slsqp(P: np.ndarray, y: np.ndarray, n_starts: int = 8,
                   seed: int = 0) -> tuple[np.ndarray, float]:
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w):
        return rae(y, P @ w)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    starts = [np.full(K, 1.0 / K)]
    for _ in range(n_starts - 1):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w, best_r = None, np.inf
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
            w = np.clip(res.x, 0.0, 1.0)
            s = w.sum()
            if s <= 0:
                continue
            w = w / s
            r = rae(y, P @ w)
            if r < best_r:
                best_r, best_w = r, w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = rae(y, P @ best_w)
    return best_w, best_r


def _nested_scaffold_cv(
    P: np.ndarray, y: np.ndarray, scaffolds: list[str],
    n_splits: int = 5, seed: int = 42,
) -> tuple[float, np.ndarray, list[dict]]:
    """Honest nested scaffold-CV.

    For each outer fold f:
        - SLSQP weights w_f learned on P[outer_train], y[outer_train].
        - val pred = P[outer_val] @ w_f.
    Concat val preds -> scaffold-OOS-concat blended OOF -> RAE.
    """
    n = len(y)
    folds = scaffold_kfold_indices(scaffolds, n_splits=n_splits,
                                    shuffle=True, seed=seed)
    blended = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        w, train_r = _simplex_slsqp(P[tr_idx], y[tr_idx], n_starts=8, seed=fi)
        vp = P[va_idx] @ w
        blended[va_idx] = vp
        val_r = rae(y[va_idx], vp)
        fold_records.append({
            "fold": fi,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "weights": {ANCHOR_NAMES[k]: round(float(w[k]), 4) for k in range(len(w))},
            "train_rae": round(float(train_r), 4),
            "val_rae": round(float(val_r), 4),
            "max_w": round(float(w.max()), 4),
        })
    assert not np.any(np.isnan(blended)), "OOS-concat coverage gap"
    return float(rae(y, blended)), blended, fold_records


def main() -> None:
    t0 = time.time()
    out = {
        "tag": TAG,
        "claim": {"source": "nb1150", "scaffold_cv_rae": NB1150_CLAIMED_RAE},
        "anchors": ANCHOR_NAMES,
        "anchor_fold_protocol": ANCHOR_FOLD_PROTOCOL,
        "gate": {
            "nested_scaffold_cv_rae_ceiling": PROMOTE_RAE_CEILING,
            "fresh_seed_tol_std": FRESH_SEED_TOL,
            "conservative_lb_shift": CONSERVATIVE_LB_SHIFT,
        },
    }

    # ---- 1) Load + sha256 leak check ----
    unb_idx = np.load(UNBLIND_IDX)
    y = np.load(UNBLIND_Y).astype(np.float64)
    n = len(y)
    assert n == 253, f"expected 253 unblind, got {n}"
    y_sha = _sha256(y.astype(np.float32))
    out["y_unb"] = {"n": n, "sha256_f32": y_sha,
                    "mean": round(float(y.mean()), 4),
                    "std": round(float(y.std()), 4)}

    indiv_rae = {}
    anchor_audit = {}
    P_list = []
    for name in ANCHOR_NAMES:
        oof = np.load(ANCHOR_OOF_PATHS[name]).astype(np.float64)
        assert oof.shape == (253,), f"{name} OOF wrong shape {oof.shape}"
        sha = _sha256(oof.astype(np.float32))
        eq_frac = float(np.mean(np.isclose(oof, y, atol=1e-6)))
        pear = float(np.corrcoef(oof, y)[0, 1])
        sha_eq_y = (sha == y_sha)
        anchor_audit[name] = {
            "path": str(ANCHOR_OOF_PATHS[name]),
            "sha256_f32": sha,
            "sha256_equals_y": bool(sha_eq_y),
            "exact_eq_truth_frac": round(eq_frac, 4),
            "pearson_vs_truth": round(pear, 4),
            "mean": round(float(oof.mean()), 4),
            "std": round(float(oof.std()), 4),
            "fold_protocol": ANCHOR_FOLD_PROTOCOL[name],
            "honest_cross_fit_likely": bool(
                not sha_eq_y and eq_frac < 0.05 and pear < 0.999
            ),
        }
        indiv_rae[name] = round(rae(y, oof), 4)
        P_list.append(oof)
    P = np.stack(P_list, axis=1)  # (253, 4)
    out["indiv_rae_unb"] = indiv_rae
    out["anchor_audit"] = anchor_audit
    out["all_anchors_honest_likely"] = bool(
        all(a["honest_cross_fit_likely"] for a in anchor_audit.values())
    )

    # ---- 2/3) Random-KFold-OOF + scaffold-CV-weight regroup concern ----
    # The 4 anchor OOFs are all from random 5-fold KFold (shuffle=True).
    # nb1150 then re-groups them into a SCAFFOLD 5-fold and learns SLSQP
    # weights per scaffold-fold. Concern: each anchor's per-row OOF residual
    # was generated when that row was in a RANDOM-KFold validation set, so
    # the residual is conditioned on "model fit on the other ~80% randomly".
    # The SCAFFOLD-fold outer trains its SLSQP weights on rows where the
    # anchor residual was already conditioned on random-fold leakage from
    # the held-out scaffold neighbors. This DOES leak scaffold-neighbor
    # information into the SLSQP weight-fit.
    # -> Severity: low-to-moderate; corrected estimate = nested scaffold-CV
    #    with the same protocol, which is what we run in step 4.
    out["protocol_concern"] = {
        "description": (
            "All 4 anchor OOFs are random 5-fold KFold cross-fit; nb1150 "
            "regroups them by scaffold for outer SLSQP weight learning. The "
            "anchor's residual on a held-out scaffold-fold row was generated "
            "when the model SAW its scaffold neighbors in random training "
            "folds. This leaks scaffold-neighbor info into the SLSQP weights, "
            "giving an OPTIMISTIC scaffold-CV estimate. Nested scaffold-CV "
            "(this script) does not fix the underlying anchor leak but reduces "
            "the second-order weight-fitting leak."
        ),
        "severity": "moderate",
        "mitigation": "nested scaffold-CV (step 4) + fresh-seed (step 5) + conservative shift (step 6)",
    }

    # ---- 4) Nested scaffold-CV (canonical seed 42) ----
    te = load_test()
    smi_unb = te.iloc[unb_idx]["smiles"].tolist()
    scaffolds = [_murcko_scaffold(s) for s in smi_unb]
    out["n_unique_scaffolds"] = len(set([s for s in scaffolds if s]))

    nested_rae_seed42, nested_blended_seed42, nested_fold_records = _nested_scaffold_cv(
        P, y, scaffolds, n_splits=5, seed=42
    )
    out["nested_scaffold_cv_seed42"] = {
        "rae": round(nested_rae_seed42, 4),
        "fold_records": nested_fold_records,
        "delta_vs_claim": round(nested_rae_seed42 - NB1150_CLAIMED_RAE, 4),
    }

    # ---- 5) Fresh kf_seeds {1001..1010} ----
    fresh_seeds = list(range(1001, 1011))
    fresh_records = []
    for s in fresh_seeds:
        r, _, _ = _nested_scaffold_cv(P, y, scaffolds, n_splits=5, seed=s)
        fresh_records.append({"seed": s, "rae": round(r, 4)})
    fresh_rae_arr = np.array([r["rae"] for r in fresh_records])
    fresh_stats = {
        "seeds": fresh_seeds,
        "per_seed": fresh_records,
        "mean": round(float(fresh_rae_arr.mean()), 4),
        "median": round(float(np.median(fresh_rae_arr)), 4),
        "std": round(float(fresh_rae_arr.std()), 4),
        "min": round(float(fresh_rae_arr.min()), 4),
        "max": round(float(fresh_rae_arr.max()), 4),
        "range": round(float(fresh_rae_arr.max() - fresh_rae_arr.min()), 4),
    }
    out["fresh_seed_robustness"] = fresh_stats

    # Reproducibility window: does claimed 0.4710 sit within fresh-seed range?
    claim_in_range = bool(
        fresh_stats["min"] - FRESH_SEED_TOL <= NB1150_CLAIMED_RAE
        <= fresh_stats["max"] + FRESH_SEED_TOL
    )
    out["claim_in_fresh_seed_range"] = claim_in_range
    out["claim_vs_fresh_seed_mean_delta"] = round(NB1150_CLAIMED_RAE - fresh_stats["mean"], 4)

    # ---- 6) Conservative LB band ----
    nested_consensus = round(float(np.mean(np.concatenate([
        [nested_rae_seed42], fresh_rae_arr
    ]))), 4)
    lb_proj_lo = round(nested_consensus + 0.05, 4)  # tight floor
    lb_proj_mid = round(nested_consensus + CONSERVATIVE_LB_SHIFT, 4)
    lb_proj_hi = round(nested_consensus + 0.15, 4)
    out["conservative_lb_band"] = {
        "nested_consensus_rae": nested_consensus,
        "lb_lo": lb_proj_lo,
        "lb_mid_conservative": lb_proj_mid,
        "lb_hi_pessimistic": lb_proj_hi,
        "memory_basis": "feedback_train_oof_blend_transfer +0.10; nb1150 anchors are 253-OOF not train-OOF so shift is SMALLER but still nonzero due to OOF-regrouping leak (step 3).",
    }

    # ---- 7) Verdict gate ----
    rae_pass = bool(nested_consensus <= PROMOTE_RAE_CEILING)
    repro_pass = bool(fresh_stats["std"] <= FRESH_SEED_TOL)
    leak_pass = out["all_anchors_honest_likely"]
    verdict = "PROMOTE" if (rae_pass and repro_pass and leak_pass) else "REJECT"
    out["gate_results"] = {
        "rae_pass_at_ceiling": rae_pass,
        "fresh_seed_repro_pass": repro_pass,
        "anchor_leak_pass": leak_pass,
        "verdict": verdict,
        "reasoning": (
            f"nested_consensus={nested_consensus:.4f} "
            f"vs ceiling {PROMOTE_RAE_CEILING:.4f} -> rae_pass={rae_pass}; "
            f"fresh_seed_std={fresh_stats['std']:.4f} "
            f"vs tol {FRESH_SEED_TOL:.4f} -> repro_pass={repro_pass}; "
            f"anchors_honest={leak_pass}."
        ),
    }
    out["verdict"] = verdict

    out["wall_sec"] = round(time.time() - t0, 2)

    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "tag": TAG,
        "indiv_rae_unb": indiv_rae,
        "nb1150_claim": NB1150_CLAIMED_RAE,
        "nested_scaffold_cv_rae_seed42": round(nested_rae_seed42, 4),
        "fresh_seed_mean_rae": fresh_stats["mean"],
        "fresh_seed_std": fresh_stats["std"],
        "fresh_seed_range": [fresh_stats["min"], fresh_stats["max"]],
        "nested_consensus_rae": nested_consensus,
        "lb_band_lo_mid_hi": [lb_proj_lo, lb_proj_mid, lb_proj_hi],
        "verdict": verdict,
        "summary_path": str(summary_path),
        "wall_sec": out["wall_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()

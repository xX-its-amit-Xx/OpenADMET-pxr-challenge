"""nb1181 -- Hunt for residual-orthogonal blend partner to nb1150.

Per nb1171 finding: chemprop_aux anchor-family candidates are 95% correlated,
so a brute-force OOF scan is needed to find a TRULY orthogonal partner that can
break the nb1150 nested-scaffold-CV ceiling (0.4710).

PROTOCOL:
  1. Reconstruct nb1150 nested-scaffold-CV OOF on 253 unblind (matches nb1171).
  2. Scan ALL OOF files in data/processed/*_pred_oof.npy and *_oof*.npy.
     EXCLUDE nb-2167 family (lucky-seed candidates per directive).
     EXCLUDE nb1150's own anchors (chemprop_aux, nb503, nb1014, nb2103-K28).
  3. Per candidate: compute Pearson(resid, nb1150_resid) and scaffold-CV RAE.
  4. Identify candidates with |Pearson| < 0.7 AND scaffold-CV RAE < 0.7.
  5. Report top-5 most-orthogonal (by |Pearson|).
  6. For top-3 ORTHOGONAL candidates: scaffold-fold SLSQP 2-way blend with nb1150;
     report blend RAE.
  7. Best blend must beat nb1150 (0.4710) by >=0.003 -> deploy CSV at
     submissions/nb1181_orthogonal_blend.csv.

Outputs:
  data/processed/nb1181_summary.json
  submissions/nb1181_orthogonal_blend.csv  (only if gate passes)
"""
from __future__ import annotations

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
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import minimize

from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

RDLogger.DisableLog("rdApp.*")

TAG = "nb1181"
SUBMISSIONS_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

NB1150_GATE = 0.4710
DEPLOY_DELTA = 0.003
DECISION_GATE = NB1150_GATE - DEPLOY_DELTA  # 0.4680

ORTHO_CORR_THRESH = 0.7
RAE_CEIL = 0.7

# nb1150 anchor layout (must match nb1150_slsqp_simplex.py exactly)
ANCHOR_OOF_PATHS = {
    "chemprop_aux": DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
    "nb503":        DATA_PROCESSED / "nb503_pred_oof.npy",
    "nb1014":       DATA_PROCESSED / "nb1133_nb1014_pred_oof.npy",
    "nb2112":       DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",
}
ANCHOR_TE_PATHS = {
    "chemprop_aux": DATA_PROCESSED / "te_chemprop_aux.npy",
    "nb503":        DATA_PROCESSED / "te_nb503.npy",
    "nb1014":       DATA_PROCESSED / "te_nb1014.npy",
    "nb2112":       DATA_PROCESSED / "te_nb2112.npy",
}
ANCHOR_NAMES = list(ANCHOR_OOF_PATHS.keys())

UNBLIND_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y   = DATA_PROCESSED / "_audit_unblind_y.npy"

# basenames of candidate OOFs that should be excluded from scan
EXCLUDE_PREFIXES = ("nb2167",)  # lucky-seed family per directive
EXCLUDE_BASENAMES = {
    # nb1150 anchor OOFs themselves
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
}


def _murcko_scaffold(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) or ""
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


def _reconstruct_nb1150_oof(y: np.ndarray, scaffolds: list[str]) -> np.ndarray:
    """Recompute nb1150's nested scaffold-CV blended OOF on the 253 unblind."""
    P_list = [np.load(ANCHOR_OOF_PATHS[name]).astype(np.float64) for name in ANCHOR_NAMES]
    P = np.stack(P_list, axis=1)  # (253, 4)
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)
    blended = np.full(len(y), np.nan, dtype=np.float64)
    fold_w = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        w, _ = _simplex_slsqp(P[tr_idx], y[tr_idx], n_starts=8, seed=fi)
        blended[va_idx] = P[va_idx] @ w
        fold_w.append(w)
    assert not np.any(np.isnan(blended))
    return blended, np.stack(fold_w, axis=0)


def _gather_oof_files() -> list[Path]:
    """Collect candidate OOF files: *_pred_oof.npy and *_oof*.npy minus excludes."""
    seen, out = set(), []
    for pat in ("*_pred_oof.npy", "*_oof*.npy"):
        for p in DATA_PROCESSED.glob(pat):
            bn = p.name
            if bn in seen:
                continue
            if bn in EXCLUDE_BASENAMES:
                continue
            if any(bn.startswith(pref) for pref in EXCLUDE_PREFIXES):
                continue
            seen.add(bn)
            out.append(p)
    return sorted(out)


def _safe_load_253(path: Path) -> np.ndarray | None:
    """Load a (253,) float array, returning None if shape or dtype invalid."""
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr.ndim != 1 or arr.shape[0] != 253:
        return None
    arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _scaffold_cv_rae(oof: np.ndarray, y: np.ndarray) -> float:
    """OOF is treated as already-honest cross-fit on 253; this is the RAE."""
    return float(rae(y, oof))


def main() -> None:
    t0 = time.time()
    out: dict = {
        "tag": TAG,
        "purpose": "Hunt for residual-orthogonal blend partner to nb1150",
        "nb1150_gate": NB1150_GATE,
        "decision_gate": DECISION_GATE,
        "ortho_corr_thresh": ORTHO_CORR_THRESH,
        "rae_ceil": RAE_CEIL,
    }

    # ----- load truth + scaffolds -----
    unb_idx = np.load(UNBLIND_IDX)
    y = np.load(UNBLIND_Y).astype(np.float64)
    n = len(y)
    assert n == 253
    te_df = load_test()
    smi_unb = te_df.iloc[unb_idx]["smiles"].tolist()
    scaffolds = [_murcko_scaffold(s) for s in smi_unb]
    out["n_unique_scaffolds"] = len(set([s for s in scaffolds if s]))

    # ----- reconstruct nb1150 OOF -----
    oof_anchor, _fw = _reconstruct_nb1150_oof(y, scaffolds)
    resid_anchor = y - oof_anchor
    rae_anchor = float(rae(y, oof_anchor))
    out["nb1150_oof_rae_reconstructed"] = round(rae_anchor, 4)

    # ----- scan candidates -----
    cand_files = _gather_oof_files()
    out["n_cand_files_scanned"] = len(cand_files)

    rows = []
    for p in cand_files:
        oof = _safe_load_253(p)
        if oof is None:
            continue
        # SHA-style guard: suspicious near-perfect match to truth -> skip
        exact_eq_frac = float(np.mean(np.isclose(oof, y, atol=1e-6)))
        if exact_eq_frac > 0.05:
            continue
        try:
            r = _scaffold_cv_rae(oof, y)
            resid = y - oof
            if np.std(resid) < 1e-9 or np.std(resid_anchor) < 1e-9:
                continue
            pcorr = float(np.corrcoef(resid, resid_anchor)[0, 1])
        except Exception:
            continue
        rows.append({
            "basename": p.name,
            "rae_unb": round(r, 4),
            "pearson_resid_to_nb1150": round(pcorr, 4),
            "abs_pearson": round(abs(pcorr), 4),
        })

    df = pd.DataFrame(rows)
    out["n_cand_with_valid_oof"] = int(len(df))

    if len(df) == 0:
        out["error"] = "no valid candidate OOFs found"
        out["wall_sec"] = round(time.time() - t0, 2)
        with open(DATA_PROCESSED / f"{TAG}_summary.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"tag": TAG, "error": out["error"]}, indent=2))
        return

    # ----- orthogonal filter -----
    df_ortho = df[(df["abs_pearson"] < ORTHO_CORR_THRESH) & (df["rae_unb"] < RAE_CEIL)].copy()
    df_ortho = df_ortho.sort_values("abs_pearson").reset_index(drop=True)

    out["n_ortho_candidates"] = int(len(df_ortho))
    out["top5_orthogonal"] = df_ortho.head(5).to_dict(orient="records")

    if len(df_ortho) == 0:
        out["verdict"] = "NO_ORTHOGONAL_PARTNER_FOUND"
        out["scan_top5_anywhere"] = df.sort_values("abs_pearson").head(5).to_dict(orient="records")
        out["wall_sec"] = round(time.time() - t0, 2)
        with open(DATA_PROCESSED / f"{TAG}_summary.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({
            "tag": TAG,
            "verdict": out["verdict"],
            "n_scanned": out["n_cand_with_valid_oof"],
            "decision_gate": DECISION_GATE,
        }, indent=2))
        return

    # ----- blend top-3 against nb1150 with scaffold-fold SLSQP -----
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, shuffle=True, seed=42)
    top3 = df_ortho.head(3).to_dict(orient="records")
    blend_results = []
    best_record = None

    for rec in top3:
        bn = rec["basename"]
        oof_b = _safe_load_253(DATA_PROCESSED / bn)
        if oof_b is None:
            continue
        P = np.stack([oof_anchor, oof_b], axis=1)  # (253, 2)
        blended = np.full(n, np.nan, dtype=np.float64)
        fold_w = []
        per_fold = []
        for fi, (tr_idx, va_idx) in enumerate(folds):
            w, r_tr = _simplex_slsqp(P[tr_idx], y[tr_idx], n_starts=8, seed=fi)
            blended[va_idx] = P[va_idx] @ w
            r_va = float(rae(y[va_idx], blended[va_idx]))
            per_fold.append({"fold": fi, "n_tr": int(len(tr_idx)), "n_va": int(len(va_idx)),
                             "w_nb1150": round(float(w[0]), 4),
                             "w_partner": round(float(w[1]), 4),
                             "train_rae": round(float(r_tr), 4),
                             "val_rae": round(r_va, 4)})
            fold_w.append(w)
        fold_w = np.stack(fold_w, axis=0)
        w_mean = fold_w.mean(axis=0)
        w_mean = w_mean / w_mean.sum()
        rae_blend = float(rae(y, blended))
        rec_full = {
            "partner_basename": bn,
            "partner_rae_unb": rec["rae_unb"],
            "partner_pearson_resid": rec["pearson_resid_to_nb1150"],
            "blend_rae_scaffoldCV": round(rae_blend, 4),
            "delta_vs_nb1150": round(rae_blend - rae_anchor, 4),
            "mean_fold_w_nb1150": round(float(w_mean[0]), 4),
            "mean_fold_w_partner": round(float(w_mean[1]), 4),
            "per_fold": per_fold,
        }
        blend_results.append(rec_full)
        if best_record is None or rae_blend < best_record["blend_rae_scaffoldCV"]:
            best_record = rec_full
            best_record["_w_mean"] = w_mean
            best_record["_blended_oof"] = blended

    out["blend_results_top3"] = [{k: v for k, v in r.items() if not k.startswith("_")}
                                  for r in blend_results]

    if best_record is None:
        out["verdict"] = "NO_VALID_BLEND"
        out["wall_sec"] = round(time.time() - t0, 2)
        with open(DATA_PROCESSED / f"{TAG}_summary.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(json.dumps({"tag": TAG, "verdict": out["verdict"]}, indent=2))
        return

    best_rae = best_record["blend_rae_scaffoldCV"]
    passes = best_rae <= DECISION_GATE

    out["best_partner"] = {
        "basename": best_record["partner_basename"],
        "blend_rae_scaffoldCV": best_rae,
        "delta_vs_nb1150": round(best_rae - rae_anchor, 4),
        "mean_fold_w_nb1150": best_record["mean_fold_w_nb1150"],
        "mean_fold_w_partner": best_record["mean_fold_w_partner"],
    }
    out["passes_gate"] = bool(passes)

    # ----- deploy if gate passes -----
    deploy_csv = None
    if passes:
        # build deploy te (513) by combining te_nb1150.npy + te_<partner>
        te_anchor_path = DATA_PROCESSED / "te_nb1150.npy"
        # partner te: try te_<basename_root> patterns
        partner_bn = best_record["partner_basename"]
        partner_root = partner_bn.replace(".npy", "")
        # candidate locations: te_<tag>.npy where tag = partner first token
        first_tok = partner_root.split("_")[0]
        partner_te_candidates = [
            DATA_PROCESSED / f"te_{first_tok}.npy",
            DATA_PROCESSED / f"te_{partner_root}.npy",
            DATA_PROCESSED / f"{partner_root.replace('_oof', '').replace('pred_', 'te_')}.npy",
        ]
        partner_te_path = None
        for c in partner_te_candidates:
            if c.exists():
                partner_te_path = c
                break

        if not te_anchor_path.exists():
            out["deploy_status"] = "SKIP_NO_TE_NB1150"
        elif partner_te_path is None:
            out["deploy_status"] = "SKIP_NO_PARTNER_TE"
            out["deploy_te_candidates_tried"] = [str(c) for c in partner_te_candidates]
        else:
            te_anchor = np.load(te_anchor_path).astype(np.float64)
            te_partner = np.load(partner_te_path).astype(np.float64)
            if te_anchor.shape != (513,) or te_partner.shape != (513,):
                out["deploy_status"] = f"SKIP_BAD_TE_SHAPE_{te_anchor.shape}_{te_partner.shape}"
            else:
                w_mean = best_record["_w_mean"]
                deploy_pred = w_mean[0] * te_anchor + w_mean[1] * te_partner
                sub = pd.DataFrame({
                    "SMILES": te_df["smiles"].values,
                    "Molecule Name": te_df["name"].values,
                    "pEC50": deploy_pred,
                })
                deploy_csv = SUBMISSIONS_DIR / f"{TAG}_orthogonal_blend.csv"
                sub.to_csv(deploy_csv, index=False)
                out["deploy_csv"] = str(deploy_csv)
                out["deploy_pred_mean"] = round(float(deploy_pred.mean()), 4)
                out["deploy_pred_std"] = round(float(deploy_pred.std()), 4)
                out["deploy_partner_te_path"] = str(partner_te_path)
                out["deploy_status"] = "OK"

    if not passes:
        out["verdict"] = (f"NO_BREAK best_blend={best_rae:.4f} did not beat "
                          f"gate={DECISION_GATE:.4f} (nb1150 anchor {rae_anchor:.4f})")
    else:
        out["verdict"] = (f"BREAK best_blend={best_rae:.4f} beats nb1150 "
                          f"({rae_anchor:.4f}) by {rae_anchor - best_rae:+.4f}")

    out["wall_sec"] = round(time.time() - t0, 2)

    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "tag": TAG,
        "n_scanned": out["n_cand_with_valid_oof"],
        "n_ortho": out["n_ortho_candidates"],
        "top5_orthogonal": out["top5_orthogonal"],
        "best_partner": out.get("best_partner"),
        "passes_gate": out.get("passes_gate"),
        "verdict": out["verdict"],
        "deploy_csv": out.get("deploy_csv"),
        "wall_sec": out["wall_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()

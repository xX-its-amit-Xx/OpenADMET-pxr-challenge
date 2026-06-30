"""nb1213 -- Shallow residual-LGBM bag on nb1070 anchor, 881-bit substructure
dictionary fingerprints features only.

Hypothesis:
    PubChem 881-bit fingerprints are another curated substructure dictionary
    (similar to MACCS-166 but ~5x bigger).  The larger curated dictionary may
    catch substructure patterns MACCS-166 misses.

Fingerprint backend (selection priority):
  (a) openbabel / pybel native PubChem fingerprint        -> NOT AVAILABLE in
      this env (openbabel module missing).
  (b) pychem / chemprop PubChem fingerprint                -> NOT AVAILABLE.
  (c) FALLBACK: rdkit RDKFingerprint(maxPath=7, fpSize=881).  This is a
      path-based hashed fingerprint (Daylight-style), structurally a
      different family from MACCS / Morgan / AtomPair, with the
      same 881-bit width as PubChem.  Documented substitution: the FEATURE
      FAMILY is NOT identical to PubChem's curated dictionary, but it is a
      different-style 881-bit fingerprint and serves as the closest
      substitute available in this pure-rdkit environment.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Anchor = nb1070 pred_oof (constant across seeds).
  2. residual = y_unb - nb1070_oof
  3. KFold(n=5, shuffle=True, random_state=s) on 253 unblind rows.
  4. Shallow LGBM Huber (max_depth=3, num_leaves=7, n_est=80, lr=0.05,
     min_child_samples=20, alpha=1.0) on the 881-bit feature matrix sliced
     to unblind.
  5. pred_corrected_s = nb1070_oof + residual_oof_s; pooled RAE.

Mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds of pred_corr_s).
Compare to nb1070 anchor (0.5771) and to nb1183 ref (0.5513) at 0.003 margin.

Orthogonality probe: Pearson correlation between this mean-bag corrected OOF
and the mean-bag corrected OOF of nb1183 (MACCS-167) and nb1070 anchor.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/te_pubchem881.npy                   (513, 881) uint8 cache
  data/processed/nb1213_per_seed_corrected_oof.npy   (5, 253)   float32
  data/processed/nb1213_mean_bag_oof.npy             (253,)     float32
  data/processed/nb1213_median_bag_oof.npy           (253,)     float32
  data/processed/nb1213_summary.json
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1213"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

PUBCHEM_TE_PATH = DATA_PROCESSED / "te_pubchem881.npy"   # (513, 881) uint8

NB1070_REF_POOLED = 0.5771
NB1183_MEAN_BAG_REF = 0.5513   # MACCS-167 residual bag on nb1070
DECISION_MARGIN = 0.003

# Fingerprint backend selection
FP_BACKEND = None  # set in _select_fp_backend()
FP_BACKEND_NOTE = None


def _select_fp_backend() -> tuple[str, str]:
    """Try (a) openbabel pybel PubChem FP, (b) pychem, (c) rdkit RDKFingerprint
    fallback.  Returns (backend_name, descriptive_note).
    """
    # (a) openbabel
    try:
        from openbabel import pybel  # noqa: F401
        return ("openbabel_pubchem", "openbabel.pybel native PubChem FP")
    except Exception:
        pass
    # (b) pychem (mcs / cdk-style)
    try:
        import pychem  # noqa: F401
        return ("pychem_pubchem", "pychem.pubchem dictionary FP")
    except Exception:
        pass
    # (c) rdkit RDKFingerprint fallback at width=881
    try:
        from rdkit.Chem import RDKFingerprint  # noqa: F401
        return (
            "rdkit_rdkfp_maxpath7_881",
            "FALLBACK: rdkit RDKFingerprint(maxPath=7, fpSize=881) "
            "-- path-based hashed FP, NOT PubChem's curated dictionary; "
            "same width 881 but different feature family.",
        )
    except Exception as e:
        raise RuntimeError(
            f"No fingerprint backend available (openbabel/pychem/rdkit): {e}"
        )


def _build_te_pubchem881(smiles_te: list[str]) -> np.ndarray:
    """Compute 881-bit fingerprints for all 513 test SMILES via the selected
    backend (FP_BACKEND).  Returns (513, 881) uint8.
    """
    n = len(smiles_te)
    X = np.zeros((n, 881), dtype=np.uint8)

    if FP_BACKEND == "openbabel_pubchem":
        from openbabel import pybel
        for i, smi in enumerate(smiles_te):
            mol = pybel.readstring("smi", smi)
            fp = mol.calcfp("pubchem")
            # pybel fp.bits is 1-indexed list of set bits
            for b in fp.bits:
                if 1 <= b <= 881:
                    X[i, b - 1] = 1
        return X

    if FP_BACKEND == "pychem_pubchem":
        # Best-effort hook; if pychem ever lands, refit here.
        raise RuntimeError("pychem backend reached but not implemented; rerun.")

    # (c) rdkit RDKFingerprint fallback
    from rdkit import Chem
    from rdkit.Chem import RDKFingerprint
    from rdkit.DataStructs import ConvertToNumpyArray

    n_fail = 0
    for i, smi in enumerate(smiles_te):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_fail += 1
            continue
        fp = RDKFingerprint(mol, maxPath=7, fpSize=881)
        arr = np.zeros(881, dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        X[i] = arr
    if n_fail:
        print(f"[fp][warn] {n_fail}/{n} SMILES failed to parse -> zero rows")
    return X


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _orthogonality_probe(mean_bag_oof: np.ndarray) -> dict:
    out = {}
    refs = {
        "nb1183": DATA_PROCESSED / "nb1183_mean_bag_oof.npy",
        "nb1070_anchor": DATA_PROCESSED / "nb1070_pred_oof.npy",
    }
    for ref_tag, p in refs.items():
        if not p.exists():
            out[f"pearson_vs_{ref_tag}"] = None
            out[f"{ref_tag}_probe_error"] = f"missing {p}"
            continue
        try:
            ref = np.load(p).astype(np.float64)
            if ref.shape[0] != mean_bag_oof.shape[0]:
                out[f"pearson_vs_{ref_tag}"] = None
                out[f"{ref_tag}_probe_error"] = (
                    f"shape mismatch: ref={ref.shape} vs "
                    f"self={mean_bag_oof.shape}"
                )
                continue
            a = mean_bag_oof.astype(np.float64)
            if a.std() > 0 and ref.std() > 0:
                r = float(np.corrcoef(a, ref)[0, 1])
            else:
                r = float("nan")
            out[f"pearson_vs_{ref_tag}"] = r
        except Exception as e:
            out[f"pearson_vs_{ref_tag}"] = None
            out[f"{ref_tag}_probe_error"] = repr(e)
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SHALLOW residual-LGBM bag on top of nb1070, "
          f"881-bit substructure FP, {len(RESID_SEEDS)} KFold seeds")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          LGBM: max_depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Build or load 881-bit fingerprint cache ----
    global FP_BACKEND, FP_BACKEND_NOTE
    FP_BACKEND, FP_BACKEND_NOTE = _select_fp_backend()
    print(f"[fp] backend = {FP_BACKEND}")
    print(f"[fp] note    = {FP_BACKEND_NOTE}")

    if PUBCHEM_TE_PATH.exists():
        X_te_all = np.load(PUBCHEM_TE_PATH)
        if X_te_all.shape != (n_test, 881):
            print(f"[fp][warn] cached shape {X_te_all.shape} unexpected; "
                  f"recomputing")
            X_te_all = _build_te_pubchem881(te["smiles"].tolist())
            np.save(PUBCHEM_TE_PATH, X_te_all)
        else:
            print(f"[fp] cached -> {PUBCHEM_TE_PATH}  shape={X_te_all.shape}")
    else:
        X_te_all = _build_te_pubchem881(te["smiles"].tolist())
        np.save(PUBCHEM_TE_PATH, X_te_all)
        print(f"[fp] saved -> {PUBCHEM_TE_PATH}  shape={X_te_all.shape}")

    X_unb = X_te_all[unb_idx].astype(np.float32)
    print(f"[feat] X_unb shape = {X_unb.shape}  (881-bit FP)")
    print(f"[feat] bit density (unb) = {X_unb.mean():.4f}  "
          f"const cols = {int((X_unb.var(axis=0) == 0).sum())}/{X_unb.shape[1]}")

    # ---- Anchor ----
    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor OOF "
            f"(run nb1070 first)."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: {anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy  pooled RAE = {rae_anchor:.4f}  "
          f"(ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow, 881-bit FP)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1183 mean_bag ref    = {NB1183_MEAN_BAG_REF:.4f}  "
          f"(MACCS-167 residual on nb1070)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_MEAN_BAG_REF - DECISION_MARGIN

    if beats_nb1183:
        verdict = "PUBCHEM881_RESIDUAL_BEATS_NB1183_LARGER_DICTIONARY_WINS"
    elif beats_nb1070:
        verdict = "PUBCHEM881_RESIDUAL_HELPS_NB1070_BUT_NOT_NB1183"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "PUBCHEM881_RESIDUAL_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "PUBCHEM881_RESIDUAL_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Orthogonality probe ----
    print("\n" + "-" * 78)
    print("ORTHOGONALITY PROBE (corrected mean-bag OOF vs prior families)")
    print("-" * 78)
    ortho = _orthogonality_probe(mean_bag_oof)
    for k, v in ortho.items():
        if isinstance(v, float):
            print(f"   {k} = {v:+.4f}")
        else:
            print(f"   {k} = {v}")

    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "fingerprint_backend": FP_BACKEND,
        "fingerprint_backend_note": FP_BACKEND_NOTE,
        "feature_dim": int(X_unb.shape[1]),
        "fp_cache_path": str(PUBCHEM_TE_PATH),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_MEAN_BAG_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "orthogonality_probe": ortho,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("fingerprint_backend", "fingerprint_backend_note",
              "rae_anchor_nb1070", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median", "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb1070",
              "delta_mean_bag_vs_nb1183",
              "beats_nb1070", "beats_nb1183",
              "verdict", "orthogonality_probe"):
        print(f"  {k}: {res.get(k)}")

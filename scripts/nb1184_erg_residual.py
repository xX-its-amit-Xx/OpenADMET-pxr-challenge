"""nb1184 -- Shallow residual-LGBM bag on nb1070 anchor, ErG (Extended Reduced
Graph / pharmacophore reduced graph) fingerprint features only.

Hypothesis:
    ErG (Extended Reduced Graph) fingerprints encode pharmacophore-level
    features -- donor / acceptor / hydrophobic / aromatic centers, plus
    topological distance between those centers.  This pharmacophore
    representation should encode pose-relevant information that 2D
    circular (Morgan), path (AtomPair), and key-based (MACCS / Avalon)
    fingerprints miss.  In particular, PXR LBD binding is pharmacophore-
    driven (large 1300 Angstrom-cubed hydrophobic pocket with sparse polar
    contacts), so pharmacophore-distance features should be especially
    relevant.

    If a shallow residual-LGBM bag on ErG of (truth - nb1070_oof) cross-fits
    to RAE meaningfully below nb1153 (Mordred residual mean-bag = 0.5640)
    at the 0.003 margin, we have a genuinely orthogonal residual source.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Anchor = nb1070 pred_oof (constant across seeds).
  2. residual = y_unb - nb1070_oof
  3. KFold(n=5, shuffle=True, random_state=s) on 253 unblind rows.
  4. Shallow LGBM Huber (max_depth=3, num_leaves=7, n_est=80, lr=0.05,
     min_child_samples=20, alpha=1.0) on ErG features sliced to unblind.
  5. pred_corrected_s = nb1070_oof + residual_oof_s; pooled RAE.

Mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds of pred_corr_s).
Compare to nb1070 anchor (0.5771) and nb1153 Mordred residual mean-bag
ref (0.5640) at 0.003 margin.

Feature caching: data/processed/tr_erg.npy, te_erg.npy.  If the cache is
absent we (re)compute via rdkit.Chem.rdReducedGraphs.GetErGFingerprint
with default sample params.  If that path raises, fall back to MACCS keys
(rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint) -- dimension 167 --
and record which fingerprint actually ran in the summary JSON.

Orthogonality probes vs nb1153 (Mordred residual bag) and nb1172
(AtomPair residual bag) mean-bag OOFs (Pearson correlation, 253 rows).

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/tr_erg.npy                          (4139, dim) float32
  data/processed/te_erg.npy                          (513,  dim) float32
  data/processed/nb1184_per_seed_corrected_oof.npy   (5, 253)   float32
  data/processed/nb1184_mean_bag_oof.npy             (253,)     float32
  data/processed/nb1184_median_bag_oof.npy           (253,)     float32
  data/processed/nb1184_summary.json
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

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1184"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

ERG_TR_PATH = DATA_PROCESSED / "tr_erg.npy"
ERG_TE_PATH = DATA_PROCESSED / "te_erg.npy"

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771
NB1153_MEAN_BAG_REF = 0.5640   # Mordred residual bag on nb1070
NB1172_MEAN_BAG_REF = 0.5659   # AtomPair residual bag on nb1070
NB1153_MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical capacity to nb1153 / nb1172 bag."""
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


def _compute_erg_matrix(smiles_list: list[str], label: str) -> tuple[np.ndarray, str]:
    """Compute ErG fingerprint matrix; fall back to MACCS keys if ErG fails.

    Returns (X, fp_name) where fp_name in {"erg", "maccs"}.
    """
    from rdkit import Chem
    try:
        from rdkit.Chem import rdReducedGraphs
        # probe with a known-good SMILES
        probe = Chem.MolFromSmiles("CCO")
        _ = rdReducedGraphs.GetErGFingerprint(probe)
        use_erg = True
    except Exception as e:  # pragma: no cover
        print(f"[fp] ErG unavailable ({e!r}); falling back to MACCS")
        use_erg = False

    if use_erg:
        # Determine dim via first valid molecule
        dim = None
        rows = []
        n_fail = 0
        for i, s in enumerate(smiles_list):
            try:
                m = Chem.MolFromSmiles(s)
                if m is None:
                    raise ValueError("RDKit parse returned None")
                fp = np.asarray(
                    rdReducedGraphs.GetErGFingerprint(m), dtype=np.float32
                )
                if dim is None:
                    dim = len(fp)
                if len(fp) != dim:
                    raise ValueError(f"dim mismatch {len(fp)} vs {dim}")
                rows.append(fp)
            except Exception as ex:
                if dim is None:
                    # placeholder; will fill after we know dim
                    rows.append(None)
                else:
                    rows.append(np.zeros(dim, dtype=np.float32))
                n_fail += 1
                if n_fail <= 3:
                    print(f"[fp] {label} row {i} ErG failed: {ex!r}")
        # Patch any None placeholders with zeros
        if dim is None:
            raise RuntimeError(f"All ErG fingerprints failed for {label}")
        for i, r in enumerate(rows):
            if r is None:
                rows[i] = np.zeros(dim, dtype=np.float32)
        X = np.vstack(rows).astype(np.float32)
        print(f"[fp] {label} ErG ok  shape={X.shape}  failures={n_fail}")
        return X, "erg"
    else:
        from rdkit.Chem import MACCSkeys
        rows = []
        n_fail = 0
        for i, s in enumerate(smiles_list):
            try:
                m = Chem.MolFromSmiles(s)
                if m is None:
                    raise ValueError("RDKit parse returned None")
                fp = MACCSkeys.GenMACCSKeys(m)
                arr = np.zeros((167,), dtype=np.float32)
                from rdkit.DataStructs import ConvertToNumpyArray
                ConvertToNumpyArray(fp, arr)
                rows.append(arr)
            except Exception as ex:
                rows.append(np.zeros((167,), dtype=np.float32))
                n_fail += 1
                if n_fail <= 3:
                    print(f"[fp] {label} row {i} MACCS failed: {ex!r}")
        X = np.vstack(rows).astype(np.float32)
        print(f"[fp] {label} MACCS ok  shape={X.shape}  failures={n_fail}")
        return X, "maccs"


def _load_or_compute_erg() -> tuple[np.ndarray, np.ndarray, str]:
    """Load cached ErG matrices, or compute + cache them.

    Returns (X_tr, X_te, fp_name).
    """
    if ERG_TR_PATH.exists() and ERG_TE_PATH.exists():
        X_tr = np.load(ERG_TR_PATH)
        X_te = np.load(ERG_TE_PATH)
        print(f"[fp] loaded cached tr_erg.npy shape={X_tr.shape}")
        print(f"[fp] loaded cached te_erg.npy shape={X_te.shape}")
        # We trust the cache was made with ErG (the script we shipped uses
        # ErG; fallback would have a different filename).  But record dim.
        fp_name = "erg" if X_tr.shape[1] == 315 else (
            "maccs" if X_tr.shape[1] == 167 else "unknown"
        )
        return X_tr.astype(np.float32), X_te.astype(np.float32), fp_name

    print(f"[fp] cache miss; computing ErG fingerprints from scratch")
    tr_df = load_train()
    te_df = load_test()
    tr_smi = tr_df["smiles"].tolist()
    te_smi = te_df["smiles"].tolist()
    X_tr, fp_tr = _compute_erg_matrix(tr_smi, "train")
    X_te, fp_te = _compute_erg_matrix(te_smi, "test")
    if fp_tr != fp_te:
        raise RuntimeError(
            f"fingerprint family mismatch between train ({fp_tr}) and test ({fp_te})"
        )
    np.save(ERG_TR_PATH, X_tr)
    np.save(ERG_TE_PATH, X_te)
    print(f"[fp] saved {ERG_TR_PATH}  shape={X_tr.shape}")
    print(f"[fp] saved {ERG_TE_PATH}  shape={X_te.shape}")
    return X_tr, X_te, fp_tr


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SHALLOW residual-LGBM bag on top of nb1070, "
          f"ErG pharmacophore reduced-graph features, "
          f"{len(RESID_SEEDS)} KFold seeds")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          features = ErG (rdReducedGraphs.GetErGFingerprint, "
          f"default sample params)")
    print(f"          LGBM: max_depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor OOF "
            f"(run nb1070 first)."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: "
            f"{anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    print(f"[feat] loading / computing ErG fingerprint matrices...")
    X_tr, X_te, fp_name = _load_or_compute_erg()
    if X_te.shape[0] != n_test:
        raise ValueError(
            f"ErG test cache shape mismatch: {X_te.shape} vs n_test={n_test}"
        )
    X_unb = X_te[unb_idx].astype(np.float32)
    print(f"[feat] X_unb shape = {X_unb.shape}  (fp_name={fp_name})")
    print(f"[feat] mean value (unb) = {X_unb.mean():.4f}  "
          f"const cols = {int((X_unb.var(axis=0) == 0).sum())}/{X_unb.shape[1]}")

    # ---- Orthogonality probe vs nb1153 (Mordred residual) and nb1172 (AtomPair residual) ----
    print("\n" + "-" * 78)
    print("ORTHOGONALITY SANITY CHECK -- OOF correlation vs Mordred + AtomPair residual bags")
    print("-" * 78)
    ortho = {}
    for ref_tag in ("nb1153", "nb1172"):
        p = DATA_PROCESSED / f"{ref_tag}_mean_bag_oof.npy"
        if p.exists():
            ref_oof = np.load(p).astype(np.float64)
            if ref_oof.shape[0] != n_unb:
                ortho[f"pearson_vs_{ref_tag}_oof"] = None
                ortho[f"{ref_tag}_oof_shape_error"] = str(ref_oof.shape)
            else:
                # Stored placeholder -- final pearson is computed against our
                # mean_bag_oof after the cross-fit below.
                ortho[f"_ref_{ref_tag}_oof_loaded"] = True
        else:
            ortho[f"pearson_vs_{ref_tag}_oof"] = None
            ortho[f"{ref_tag}_oof_missing_path"] = str(p)
    for k, v in ortho.items():
        print(f"   {k} = {v}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow, {fp_name.upper()} "
          f"dim={X_unb.shape[1]})")
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
    print(f"   nb1153 mean_bag ref    = {NB1153_MEAN_BAG_REF:.4f}  "
          f"(Mordred residual on nb1070)")
    print(f"   nb1172 mean_bag ref    = {NB1172_MEAN_BAG_REF:.4f}  "
          f"(AtomPair residual on nb1070)")

    beats_nb1070 = rae_mean_bag < rae_anchor - NB1153_MARGIN
    beats_nb1172 = rae_mean_bag < NB1172_MEAN_BAG_REF - NB1153_MARGIN
    beats_nb1153 = rae_mean_bag < NB1153_MEAN_BAG_REF - NB1153_MARGIN

    if beats_nb1153:
        verdict = "ERG_RESIDUAL_BEATS_NB1153_PHARMACOPHORE_SIGNAL_REAL"
    elif beats_nb1172:
        verdict = "ERG_RESIDUAL_BEATS_NB1172_BUT_NOT_NB1153"
    elif beats_nb1070:
        verdict = "ERG_RESIDUAL_HELPS_NB1070_BUT_NOT_NB1153"
    elif abs(rae_mean_bag - rae_anchor) < NB1153_MARGIN:
        verdict = "ERG_RESIDUAL_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "ERG_RESIDUAL_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Final orthogonality: pearson of our mean_bag_oof vs nb1153 / nb1172 ----
    print("\n" + "-" * 78)
    print("MEAN-BAG OOF PEARSON vs nb1153 / nb1172")
    print("-" * 78)
    for ref_tag in ("nb1153", "nb1172"):
        p = DATA_PROCESSED / f"{ref_tag}_mean_bag_oof.npy"
        if p.exists():
            ref_oof = np.load(p).astype(np.float64)
            if ref_oof.shape[0] == n_unb:
                r = _pearson(mean_bag_oof, ref_oof)
                ortho[f"pearson_vs_{ref_tag}_mean_bag_oof"] = r
                print(f"   pearson(nb1184_mean_bag, {ref_tag}_mean_bag) = {r:+.4f}")
            else:
                ortho[f"pearson_vs_{ref_tag}_mean_bag_oof"] = None
        else:
            ortho[f"pearson_vs_{ref_tag}_mean_bag_oof"] = None

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
        "feature_source": f"{fp_name}_pharmacophore_reduced_graph",
        "fp_used": fp_name,
        "erg_cache_train": str(ERG_TR_PATH),
        "erg_cache_test": str(ERG_TE_PATH),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
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
        "delta_median_bag_vs_nb1070": rae_median_bag - rae_anchor,
        "delta_mean_bag_vs_nb1153": rae_mean_bag - NB1153_MEAN_BAG_REF,
        "delta_mean_bag_vs_nb1172": rae_mean_bag - NB1172_MEAN_BAG_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1153": bool(beats_nb1153),
        "beats_nb1172": bool(beats_nb1172),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
        "nb1172_mean_bag_ref": NB1172_MEAN_BAG_REF,
        "decision_margin": NB1153_MARGIN,
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
    for k in ("fp_used", "feature_dim",
              "rae_anchor_nb1070", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb1070",
              "delta_mean_bag_vs_nb1153",
              "delta_mean_bag_vs_nb1172",
              "beats_nb1070", "beats_nb1153", "beats_nb1172",
              "verdict", "orthogonality_probe"):
        print(f"  {k}: {res.get(k)}")

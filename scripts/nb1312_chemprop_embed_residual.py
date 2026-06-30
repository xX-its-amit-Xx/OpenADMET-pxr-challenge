"""nb1312 -- Chemprop encoder MPNN embeddings as residual feature.

Hypothesis:
    chemprop_aux was trained as a multitask MPNN on PXR + 7 NR auxiliary
    targets (BondMessagePassing depth=3, d_h=300).  The pre-FFN molecule
    embedding (300-dim, post-aggregation) is a PXR-task-conditioned chemical
    representation that the LGBM/MACCS family cannot capture.  Concatenated
    with MACCS-167 (467 cols total), a shallow residual-LGBM bag on the
    nb1070 anchor may pick up signal orthogonal to MACCS alone (nb1183) and
    to the kNN counter-feature variant (nb1242).

Pipeline:
    1. Load 5 chemprop_aux fold checkpoints (BAD4141-trained, multitask).
    2. For each fold: forward 513 test SMILES through (message_passing + agg)
       -> 300-dim mol embedding.  Average across folds for stability.
    3. Cache te_chemprop_embed_300.npy (513, 300); slice to 253 unblind.
    4. Anchor = nb1070_pred_oof.  Features = [MACCS-167 | embed-300] (467).
    5. 5-seed bag (seeds 0/1/7/42/137) shallow LGBM Huber (depth=3, leaves=7,
       80 est, lr=0.05, min_child_samples=20, alpha=1.0).  Per-seed 5-fold
       KFold cross-fit residual on (y_unb - nb1070_oof); pred_corr_s =
       anchor + resid_oof_s; pooled RAE.
    6. Mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds).
    7. Verdict at 0.003 margin vs nb1070 / nb1183 / nb1242.

Honest 253-only cross-fit diagnostic.  NO deploy refit.

Outputs:
    data/processed/te_chemprop_embed_300.npy            (513, 300) float32
    data/processed/nb1312_per_seed_corrected_oof.npy    (5, 253)   float32
    data/processed/nb1312_mean_bag_oof.npy              (253,)     float32
    data/processed/nb1312_median_bag_oof.npy            (253,)     float32
    data/processed/nb1312_summary.json
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

TAG = "nb1312"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
EMBED_CACHE = DATA_PROCESSED / "te_chemprop_embed_300.npy"

CHEMPROP_FOLD_DIRS = [
    DATA_PROCESSED / f"chemprop_fold_{i}" / "model" / "model_0" / "checkpoints"
    for i in range(5)
]

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771
NB1183_MEAN_BAG_REF = 0.5640
NB1242_MEAN_BAG_REF = 0.5640   # nb1242 ChEMBL-kNN counter-feature anchor
DECISION_MARGIN = 0.003


def _find_best_ckpt(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    cands = sorted(ckpt_dir.glob("best-*.ckpt"))
    if cands:
        return cands[0]
    last = ckpt_dir / "last.ckpt"
    return last if last.exists() else None


def _extract_embeddings(test_smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    """Extract 300-dim mol embeddings from each fold's MPN encoder, average.

    Returns (embed (n_test, 300) float32, ckpt_paths_used).
    """
    import torch
    from chemprop.models import MPNN
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.data.collate import collate_batch

    fold_ckpts: list[Path] = []
    for d in CHEMPROP_FOLD_DIRS:
        ck = _find_best_ckpt(d)
        if ck is not None:
            fold_ckpts.append(ck)
    if not fold_ckpts:
        raise FileNotFoundError(
            "No chemprop_aux fold checkpoints found under "
            f"{CHEMPROP_FOLD_DIRS[0].parent.parent}"
        )
    print(f"[embed] {len(fold_ckpts)} fold checkpoints located")
    for ck in fold_ckpts:
        print(f"        - {ck}")

    # Build dataset once; SMILES are read by chemprop's RDKit parser.
    print(f"[embed] building MoleculeDataset for {len(test_smiles)} test SMILES")
    dps = [MoleculeDatapoint.from_smi(s) for s in test_smiles]
    ds = MoleculeDataset(dps)

    # Batch in chunks to keep memory bounded on CPU.
    chunk = 64
    n = len(ds)
    accum = np.zeros((n, 300), dtype=np.float64)

    for fi, ckpt_path in enumerate(fold_ckpts):
        t0 = time.time()
        mdl = MPNN.load_from_checkpoint(str(ckpt_path), map_location="cpu")
        mdl.eval()
        d_h = int(mdl.message_passing.output_dim)
        if d_h != 300:
            raise RuntimeError(
                f"fold {fi}: expected mp.output_dim=300, got {d_h}"
            )
        with torch.no_grad():
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                batch = collate_batch([ds[j] for j in range(start, end)])
                bmg = batch.bmg
                H = mdl.message_passing(bmg, V_d=None)
                h_mol = mdl.agg(H, bmg.batch)
                accum[start:end] += h_mol.cpu().numpy().astype(np.float64)
        del mdl
        print(f"[embed] fold {fi}: forwarded {n} mols in "
              f"{time.time()-t0:.1f}s")

    embed = (accum / float(len(fold_ckpts))).astype(np.float32)
    return embed, [str(p) for p in fold_ckpts]


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
    for ref_tag in ("nb1183", "nb1242"):
        p = DATA_PROCESSED / f"{ref_tag}_mean_bag_oof.npy"
        if not p.exists():
            out[f"pearson_vs_{ref_tag}_mean_bag"] = None
            out[f"{ref_tag}_probe_error"] = f"missing {p}"
            continue
        try:
            ref = np.load(p).astype(np.float64)
            if ref.shape[0] != mean_bag_oof.shape[0]:
                out[f"pearson_vs_{ref_tag}_mean_bag"] = None
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
            out[f"pearson_vs_{ref_tag}_mean_bag"] = r
        except Exception as e:
            out[f"pearson_vs_{ref_tag}_mean_bag"] = None
            out[f"{ref_tag}_probe_error"] = repr(e)
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- chemprop encoder MPN embeddings (300-d) as residual "
          f"feature, concatenated with MACCS-167 (467 cols total).")
    print(f"          residual learner: 5-seed shallow LGBM Huber bag")
    print(f"          seeds = {RESID_SEEDS}  folds = {RESID_FOLDS}")
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
            f"{anchor_path} not found; required anchor OOF."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # ---- Chemprop encoder embeddings ----
    ckpt_paths_used: list[str] = []
    n_folds_used = 0
    embed_dim = 0
    encoder_status = "ok"

    if EMBED_CACHE.exists():
        embed_te = np.load(EMBED_CACHE)
        if embed_te.shape[0] != n_test:
            print(f"[embed] cached shape {embed_te.shape} mismatches "
                  f"n_test={n_test}; recomputing")
            embed_te, ckpt_paths_used = _extract_embeddings(te["smiles"].tolist())
            np.save(EMBED_CACHE, embed_te)
        else:
            print(f"[embed] loaded cached {EMBED_CACHE} shape={embed_te.shape}")
            # fold ckpts still discovered for reporting
            for d in CHEMPROP_FOLD_DIRS:
                ck = _find_best_ckpt(d)
                if ck is not None:
                    ckpt_paths_used.append(str(ck))
    else:
        try:
            embed_te, ckpt_paths_used = _extract_embeddings(te["smiles"].tolist())
        except FileNotFoundError as e:
            print(f"[embed] SKIP: {e}")
            summary = {
                "tag": TAG,
                "status": "chemprop_no_ckpt",
                "reason": str(e),
                "wall_sec": round(time.time() - t0, 2),
            }
            out_path = DATA_PROCESSED / f"{TAG}_summary.json"
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[save] {out_path}")
            return summary
        np.save(EMBED_CACHE, embed_te)
        print(f"[save] {EMBED_CACHE}  shape={embed_te.shape}")

    n_folds_used = len(ckpt_paths_used)
    embed_dim = int(embed_te.shape[1])

    embed_unb = embed_te[unb_idx].astype(np.float32)
    print(f"[embed] unb slice shape={embed_unb.shape}  "
          f"mean={embed_unb.mean():.4f}  std={embed_unb.std():.4f}")

    # ---- MACCS-167 ----
    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(
            f"MACCS test cache shape {X_maccs_te.shape} != n_test={n_test}"
        )
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    print(f"[feat] MACCS unb shape = {X_maccs_unb.shape}")

    X_unb = np.concatenate([X_maccs_unb, embed_unb], axis=1).astype(np.float32)
    print(f"[feat] combined X_unb shape = {X_unb.shape}  "
          f"({X_maccs_unb.shape[1]} MACCS + {embed_unb.shape[1]} embed)")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (shallow, MACCS+embed {X_unb.shape[1]})")
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
          f"(MACCS-only residual on nb1070)")
    print(f"   nb1242 mean_bag ref    = {NB1242_MEAN_BAG_REF:.4f}  "
          f"(ChEMBL-kNN counter-feature on nb1070)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_MEAN_BAG_REF - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_MEAN_BAG_REF - DECISION_MARGIN

    if beats_nb1183 and beats_nb1242:
        verdict = "CHEMPROP_EMBED_BEATS_NB1183_AND_NB1242_NEW_RESIDUAL_LANE"
    elif beats_nb1183:
        verdict = "CHEMPROP_EMBED_BEATS_NB1183_NOT_NB1242"
    elif beats_nb1242:
        verdict = "CHEMPROP_EMBED_BEATS_NB1242_NOT_NB1183"
    elif beats_nb1070:
        verdict = "CHEMPROP_EMBED_HELPS_NB1070_BUT_NOT_PRIOR_RESIDUALS"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "CHEMPROP_EMBED_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "CHEMPROP_EMBED_HURTS_NB1070"
    print(f"   verdict                = {verdict}")

    # ---- Orthogonality probe ----
    print("\n" + "-" * 78)
    print("ORTHOGONALITY PROBE (corrected mean-bag OOF vs nb1183, nb1242)")
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
        "status": "ok",
        "anchor": ANCHOR,
        "feature_source": "maccs_167_plus_chemprop_aux_embed_300",
        "embed_dim": embed_dim,
        "encoder_ckpts_used": ckpt_paths_used,
        "n_folds_encoder": n_folds_used,
        "encoder_status": encoder_status,
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
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_MEAN_BAG_REF,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_MEAN_BAG_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "nb1242_mean_bag_ref": NB1242_MEAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "orthogonality_probe": ortho,
        "embed_cache_path": str(EMBED_CACHE),
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
    for k in ("status", "encoder_ckpts_used", "embed_dim",
              "rae_anchor_nb1070", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb1070",
              "delta_mean_bag_vs_nb1183",
              "delta_mean_bag_vs_nb1242",
              "beats_nb1070", "beats_nb1183", "beats_nb1242",
              "verdict", "orthogonality_probe"):
        print(f"  {k}: {res.get(k)}")

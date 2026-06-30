"""nb3113 -- Per-row anchor selection by max Tanimoto-to-train.

NEW PARADIGM:
    Prior per-row blends (nb2933) traded K-anchor variants against each
    other (in-manifold vs deep-pyramid). All such variants live on the SAME
    chemprop_aux substrate, so per-row gating only reshuffles within
    paradigm-matched anchors and never escapes the 0.4718 deep-30 ceiling.

    nb3113 swaps the LOW-SIM branch out of paradigm: high-sim rows trust
    K18 (the verified deep-30 PRE-clean K-anchor ceiling 0.4536), low-sim
    rows fall back to chemprop_aux honest cross-fit (0.5879 on 253) -- a
    DIFFERENT anchor axis (chemprop GNN multitask head, no post-hoc
    LGBM-residual stack). The hypothesis is that on novel-scaffold rows
    (max Tanimoto to 4139 train < ~0.45) K18 is overfit to the
    Mordred/SHAP substrate cluster, and chemprop_aux's GNN message-passing
    generalises better off-manifold.

PROTOCOL:
    For each row in the 253 unblind:
      tan_max     = max ECFP4 Tanimoto vs 4139 train corpus
      w_K18       = 0.5 + 0.4 * sigmoid(10 * (tan_max - 0.45))
                    in [0.5, 0.9] -- K18 always carries >=50% weight
                    (so we never throw K18 away even on the most
                     off-manifold row)
      pred        = w_K18 * K18 + (1 - w_K18) * chemprop_aux_oof

    The 4139 train corpus is fixed (no fold leak risk: both K18_OOF and
    chemprop_aux_OOF are precomputed honest cross-fits on the 253, and
    the per-row weight is a deterministic function of test SMILES vs
    train SMILES -- never touches truth).

    15 fresh kf_seeds {1141..1155}: K18 and chemprop_aux OOFs are
    precomputed and constant, but the kf_seed varies the scaffold-fold
    PARTITION for per-fold val-RAE variance characterization. Pooled RAE
    equals full-array RAE for every seed (deterministic), matching the
    nb2991 protocol convention.

    Deploy te (513): use 4139 train + 253 unblind as Tanimoto corpus.

GATE:
    mean_rae < 0.4475 -> "BETTER"
    else              -> "FAIL"

INPUTS:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy            -- K18 deep-30 OOF (253,)
    data/processed/nb2960_K18_30seed_te.npy             -- K18 deep-30 te (513,)
    data/processed/nb1133_chemprop_aux_pred_oof.npy     -- chemprop_aux 253-OOF
    data/processed/te_chemprop_aux.npy                  -- chemprop_aux 513 deploy te

OUTPUTS:
    data/processed/nb3113_summary.json
    data/processed/nb3113_pred_oof.npy   (253,) float32
    data/processed/te_nb3113.npy         (513,) float32
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

import numpy as np

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb3113"

# --- inputs ---
K18_OOF_PATH = DATA_PROCESSED / "nb2960_K18_30seed_oof.npy"
K18_TE_PATH = DATA_PROCESSED / "nb2960_K18_30seed_te.npy"
CHEMPROP_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
CHEMPROP_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# --- per-row weight (K18 always >= 0.5 by construction) ---
SIGMOID_CENTER = 0.45
SIGMOID_SLOPE = 10.0
W_K18_BASE = 0.5         # floor weight on K18
W_K18_RANGE = 0.4        # adds up to +0.4 -> weight in [0.5, 0.9]

# --- CV ---
N_FOLDS = 5
KF_SEEDS = list(range(1141, 1156))   # 15 fresh kf_seeds

# --- gate ---
GATE_BETTER = 0.4475

# --- references ---
REF_K18_DEEP30 = 0.4536
REF_CHEMPROP_AUX_OOF = 0.5879     # nb1133 honest 253-CV
REF_NB2980_PRIMARY1 = 0.4535      # current PRIMARY-1
REF_NB2171_PYRAMID = 0.4682       # 5-anchor pyramid ceiling
REF_DEEP30_CEILING = 0.4718       # co-converged deep-30 post-hoc-blend ceiling


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tanimoto_max_vs_corpus(fp_query: np.ndarray, fp_corpus: np.ndarray) -> np.ndarray:
    """Vectorised max ECFP4 Tanimoto: (Q, B) uint8 query vs (C, B) uint8 corpus -> (Q,) float32."""
    q = fp_query.astype(np.uint16)
    c = fp_corpus.astype(np.uint16)
    q_pop = q.sum(axis=1).astype(np.float32)
    c_pop = c.sum(axis=1).astype(np.float32)
    inter = (q @ c.T).astype(np.float32)
    union = q_pop[:, None] + c_pop[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim.max(axis=1).astype(np.float32)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-row Tanimoto anchor select: K18 (high-sim) vs chemprop_aux (low-sim)")
    print(f"           w_K18 = {W_K18_BASE} + {W_K18_RANGE} * sigmoid({SIGMOID_SLOPE} * (tan - {SIGMOID_CENTER}))")
    print(f"           weight in [{W_K18_BASE:.1f}, {W_K18_BASE + W_K18_RANGE:.1f}]")
    print(f"           kf_seeds = {KF_SEEDS}")
    print(f"           gate: mean < {GATE_BETTER} -> BETTER")
    print("=" * 78)

    # ---- Truth + indices ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, n_unb

    # ---- Anchor OOFs ----
    oof_k18 = np.load(K18_OOF_PATH).astype(np.float64)
    oof_chem = np.load(CHEMPROP_OOF_PATH).astype(np.float64)
    te_k18 = np.load(K18_TE_PATH).astype(np.float64)
    te_chem = np.load(CHEMPROP_TE_PATH).astype(np.float64)
    assert oof_k18.shape == (n_unb,), oof_k18.shape
    assert oof_chem.shape == (n_unb,), oof_chem.shape

    base_k18 = float(rae(y_unb, oof_k18))
    base_chem = float(rae(y_unb, oof_chem))
    print(f"[base] K18 deep-30 OOF RAE       = {base_k18:.4f}  (ref {REF_K18_DEEP30})")
    print(f"[base] chemprop_aux OOF RAE      = {base_chem:.4f}  (ref {REF_CHEMPROP_AUX_OOF})")

    # ---- Load SMILES (513 test, 4139 train) ----
    te = load_test()
    tr = load_train()
    te_smiles = (te["smiles"].values if "smiles" in te.columns else te["SMILES"].values)
    te_names = (te["name"].values if "name" in te.columns else te["Molecule Name"].values)
    tr_smiles = tr["smiles"].values
    n_te = len(te_smiles)
    assert n_te == 513, n_te
    print(f"[load] te=513 train={len(tr_smiles)}")

    unb_smiles = [te_smiles[i] for i in unb_idx]

    # ---- Morgan ECFP4 (2048 bits) ----
    print("[fp] Morgan ECFP4 for unblind-253 ...")
    fp_unb = morgan_fp_batch(unb_smiles, radius=2, n_bits=2048)
    print(f"[fp]   fp_unb shape={fp_unb.shape}  nonzero_rows={(fp_unb.sum(axis=1) > 0).sum()}")
    print("[fp] Morgan ECFP4 for train-4139 ...")
    fp_tr = morgan_fp_batch(list(tr_smiles), radius=2, n_bits=2048)
    print(f"[fp]   fp_tr shape={fp_tr.shape}  nonzero_rows={(fp_tr.sum(axis=1) > 0).sum()}")

    # ---- Per-row weights vs 4139 train corpus ONLY (no fold leak) ----
    # The Tanimoto query is the test-side row; the corpus is the FIXED 4139
    # train manifold. This is leak-free because the corpus never contains
    # truth, and K18/chemprop OOFs are precomputed honest cross-fits.
    print("[tan] computing max ECFP4 Tanimoto (unblind-253 vs train-4139) ...")
    tan_max_unb = _tanimoto_max_vs_corpus(fp_unb, fp_tr)
    w_k18_unb = (W_K18_BASE + W_K18_RANGE * _sigmoid(SIGMOID_SLOPE * (tan_max_unb - SIGMOID_CENTER))).astype(np.float64)
    print(f"[tan] tan_max_unb: median={np.median(tan_max_unb):.3f}  "
          f"min={tan_max_unb.min():.3f}  max={tan_max_unb.max():.3f}  "
          f"mean={tan_max_unb.mean():.3f}")
    print(f"[tan] w_K18_unb:   median={np.median(w_k18_unb):.3f}  "
          f"min={w_k18_unb.min():.3f}  max={w_k18_unb.max():.3f}  "
          f"mean={w_k18_unb.mean():.3f}")
    print(f"[tan] frac tan>=0.45: {(tan_max_unb >= 0.45).mean():.3f}  "
          f"frac tan>=0.55: {(tan_max_unb >= 0.55).mean():.3f}  "
          f"frac tan<0.35: {(tan_max_unb < 0.35).mean():.3f}")

    # ---- OOF blend (precomputed, constant across seeds) ----
    oof_blend = (w_k18_unb * oof_k18 + (1.0 - w_k18_unb) * oof_chem)
    pooled_rae = float(rae(y_unb, oof_blend))
    print(f"\n[blend] pooled OOF RAE (per-row K18/chemprop) = {pooled_rae:.4f}")
    print(f"[blend] delta vs K18 alone                    = {pooled_rae - base_k18:+.4f}")
    print(f"[blend] delta vs chemprop_aux alone           = {pooled_rae - base_chem:+.4f}")

    # ---- Wide-seed sweep: characterise per-fold val-RAE variance ----
    # Pooled RAE is deterministic (OOFs precomputed); seed varies fold partition.
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaf] n_unique_scaffolds_unb253 = {n_uniq_scaf}")
    print("\n" + "-" * 78)
    print(f"WIDE-SEED SWEEP: {len(KF_SEEDS)} fresh kf_seeds {{1141..1155}}")
    print("-" * 78)

    seed_records = []
    pooled_raes = []
    per_fold_means = []
    per_fold_stds = []
    for s in KF_SEEDS:
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS, seed=s)
        fold_val_raes = []
        for _tr_loc, va_loc in splits:
            fold_val_raes.append(float(rae(y_unb[va_loc], oof_blend[va_loc])))
        pooled = float(rae(y_unb, oof_blend))
        per_fold_means.append(float(np.mean(fold_val_raes)))
        per_fold_stds.append(float(np.std(fold_val_raes, ddof=1)))
        pooled_raes.append(pooled)
        seed_records.append({
            "kf_seed": int(s),
            "pooled_rae": round(pooled, 4),
            "per_fold_val_rae_mean": round(float(np.mean(fold_val_raes)), 4),
            "per_fold_val_rae_std": round(float(np.std(fold_val_raes, ddof=1)), 4),
            "per_fold_val_rae_min": round(float(np.min(fold_val_raes)), 4),
            "per_fold_val_rae_max": round(float(np.max(fold_val_raes)), 4),
        })
        print(f"   kf={s}: pooled={pooled:.4f}  "
              f"fold_mean={np.mean(fold_val_raes):.4f}  "
              f"fold_std={np.std(fold_val_raes, ddof=1):.4f}  "
              f"[{np.min(fold_val_raes):.4f}, {np.max(fold_val_raes):.4f}]")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = std_rae / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
    t_mult = 2.145  # n=15 df=14
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))
    fold_mean_arr = np.asarray(per_fold_means)
    fold_std_arr = np.asarray(per_fold_stds)

    print("\n" + "-" * 78)
    print("AGGREGATE (15 fresh seeds)")
    print("-" * 78)
    print(f"   pooled mean    = {mean_rae:.4f}")
    print(f"   pooled std     = {std_rae:.4f}  (NOTE: ~0 expected -- OOFs precomputed)")
    print(f"   95% CI         = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median         = {median_rae:.4f}")
    print(f"   per-fold mean across seeds = "
          f"{fold_mean_arr.mean():.4f} +/- {fold_mean_arr.std(ddof=1):.4f}")
    print(f"   per-fold std  across seeds = "
          f"{fold_std_arr.mean():.4f} +/- {fold_std_arr.std(ddof=1):.4f}")

    # ---- Deploy te (513) ----
    print("\n[deploy] Morgan ECFP4 for 513 test ...")
    fp_te = morgan_fp_batch(list(te_smiles), radius=2, n_bits=2048)
    print(f"[deploy] fp_te shape={fp_te.shape}")
    # Corpus for deploy: 4139 train + 253 unblind (full known manifold)
    deploy_corpus = np.vstack([fp_tr, fp_unb])
    tan_max_te = _tanimoto_max_vs_corpus(fp_te, deploy_corpus)
    w_k18_te = (W_K18_BASE + W_K18_RANGE * _sigmoid(SIGMOID_SLOPE * (tan_max_te - SIGMOID_CENTER))).astype(np.float64)
    te_blend = (w_k18_te * te_k18 + (1.0 - w_k18_te) * te_chem).astype(np.float32)
    te_blend = np.clip(te_blend, 3.0, 9.0)
    print(f"[deploy] tan_max_te median={np.median(tan_max_te):.3f}  "
          f"w_K18_te mean={w_k18_te.mean():.3f}")
    print(f"[deploy] te_blend mean={te_blend.mean():.3f} std={te_blend.std():.3f}")
    te_unb_in_rae = float(rae(y_unb, te_blend[unb_idx]))
    print(f"[deploy] te[unb_idx] RAE (in-sample) = {te_unb_in_rae:.4f}")

    # ---- Gate ----
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
    else:
        verdict = "FAIL"
    print(f"   mean_rae = {mean_rae:.4f}  gate < {GATE_BETTER}  -> {verdict}")

    # ---- Save ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_blend.astype(np.float32))
    np.save(te_path, te_blend)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "per_row_tanimoto_anchor_select_K18_vs_chemprop_aux",
        "anchors": ["K18_deep30", "chemprop_aux_nb1133_oof"],
        "anchor_pre_unblind": True,
        "anchor_paradigm_high_sim": "K18_LGBM_residual_stack_chemprop_aux_anchor",
        "anchor_paradigm_low_sim": "chemprop_aux_GNN_multitask_PRE_unblind_clean",
        "sigmoid_center": SIGMOID_CENTER,
        "sigmoid_slope": SIGMOID_SLOPE,
        "w_K18_base": W_K18_BASE,
        "w_K18_range": W_K18_RANGE,
        "w_K18_min": W_K18_BASE,
        "w_K18_max": W_K18_BASE + W_K18_RANGE,
        "fp": "Morgan_ECFP4_radius=2_bits=2048",
        "corpus_oof": "train_4139_only (leak-free fixed corpus)",
        "corpus_deploy": "train_4139 + unb_253",
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "n_unique_scaffolds_unb253": int(n_uniq_scaf),
        "baseline_K18_oof_rae": round(base_k18, 4),
        "baseline_chemprop_aux_oof_rae": round(base_chem, 4),
        "pooled_rae_blend": round(pooled_rae, 4),
        "delta_vs_K18_alone": round(pooled_rae - base_k18, 4),
        "delta_vs_chemprop_aux_alone": round(pooled_rae - base_chem, 4),
        "tan_max_unb_median": float(np.median(tan_max_unb)),
        "tan_max_unb_mean": float(tan_max_unb.mean()),
        "tan_max_unb_min": float(tan_max_unb.min()),
        "tan_max_unb_max": float(tan_max_unb.max()),
        "tan_max_unb_frac_ge_0p45": float((tan_max_unb >= 0.45).mean()),
        "tan_max_unb_frac_ge_0p55": float((tan_max_unb >= 0.55).mean()),
        "tan_max_unb_frac_lt_0p35": float((tan_max_unb < 0.35).mean()),
        "w_K18_unb_mean": float(w_k18_unb.mean()),
        "w_K18_unb_median": float(np.median(w_k18_unb)),
        "w_K18_unb_min": float(w_k18_unb.min()),
        "w_K18_unb_max": float(w_k18_unb.max()),
        "tan_max_te_median": float(np.median(tan_max_te)),
        "w_K18_te_mean": float(w_k18_te.mean()),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "per_fold_mean_across_seeds_mean": round(float(fold_mean_arr.mean()), 4),
        "per_fold_mean_across_seeds_std": round(float(fold_mean_arr.std(ddof=1)), 4),
        "per_fold_std_across_seeds_mean": round(float(fold_std_arr.mean()), 4),
        "per_fold_std_across_seeds_std": round(float(fold_std_arr.std(ddof=1)), 4),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "deploy_te_mean": float(te_blend.mean()),
        "deploy_te_std": float(te_blend.std()),
        "ref_K18_deep30": REF_K18_DEEP30,
        "ref_chemprop_aux_oof": REF_CHEMPROP_AUX_OOF,
        "ref_nb2980_primary1": REF_NB2980_PRIMARY1,
        "ref_nb2171_pyramid": REF_NB2171_PYRAMID,
        "ref_deep30_ceiling": REF_DEEP30_CEILING,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (15 seeds)       = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   pooled OOF RAE (blend)    = {pooled_rae:.4f}")
    print(f"   K18 alone                 = {base_k18:.4f}")
    print(f"   chemprop_aux alone        = {base_chem:.4f}")
    print(f"   delta vs K18              = {pooled_rae - base_k18:+.4f}")
    print(f"   tan_max OOF median        = {np.median(tan_max_unb):.3f}")
    print(f"   w_K18 OOF mean            = {w_k18_unb.mean():.3f}")
    print(f"   verdict                   = {verdict}")
    print(f"   wall                      = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== KEY ====")
    for k in (
        "mean_rae", "pooled_rae_blend", "baseline_K18_oof_rae",
        "baseline_chemprop_aux_oof_rae", "delta_vs_K18_alone",
        "tan_max_unb_median", "w_K18_unb_mean", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")

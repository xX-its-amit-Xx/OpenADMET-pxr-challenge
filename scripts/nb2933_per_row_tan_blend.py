"""nb2933 -- Per-row Tanimoto-weighted blend: nb2240 (in-manifold) vs nb1191 (deep-stack).

NEW PARADIGM:
    For each held-out test row, compute its max ECFP4 Tanimoto to the
    *fold-train* manifold (4139 TRAIN + the train portion of the 253 in
    that fold). High-sim rows trust nb2240 (the K=20 RFE-tuned anchor swap
    that converges in a 0.4630 neighbourhood); low-sim rows trust nb1191
    (the deeper 4-anchor PRE-clean pyramid that better tolerates novel
    scaffolds).

    Sigmoid bridge:
        w_nb2240 = sigmoid(10 * (tan_max - 0.45))
        pred     = w_nb2240 * nb2240 + (1 - w_nb2240) * nb1191

PROTOCOL:
    * 5-fold scaffold CV on the 253 unblind, kf_seed 1001
      (matches the deep-30 verify protocol baseline).
    * Per fold:
        - Build fold-train Morgan FP corpus = train_4139_FP +
          unb_train_FP (the 80% of the 253 in that fold's training side).
        - For each val row: tan_max = max Tanimoto vs corpus -> scalar.
        - w = 1/(1+exp(-10*(tan_max-0.45))) -> [0..1].
        - oof[val] = w*nb2240_oof[val] + (1-w)*nb1191_oof[val].
    * Deploy (513) uses the FULL train manifold = 4139 train + 253 unblind
      (deploy-style refit; 513 includes the 253 indices).

GATES:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4598  -> MARGINAL_BEAT  (vs nb2240 5-seed pooled mean 0.4598)
    else               -> FAIL

OUTPUTS:
    scripts/nb2933_per_row_tan_blend.py
    data/processed/nb2933_summary.json
    data/processed/nb2933_pred_oof.npy        (253,) float32
    data/processed/te_nb2933.npy              (513,) float32
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

TAG = "nb2933"

# --- inputs ---
NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB1191_OOF_PATH = DATA_PROCESSED / "nb1191_pred_oof.npy"
TE_NB2240_PATH = DATA_PROCESSED / "te_nb2240.npy"
TE_NB1191_PATH = DATA_PROCESSED / "te_nb1191.npy"

# --- gates ---
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
NB2240_POOLED_REF = 0.4598  # nb2240 5-seed pooled mean (from nb2240_summary.json)
NB1191_OOF_REF = 0.4703     # nb1191 5-seed pooled mean (from nb1191_summary.json)

# --- per-row blend params ---
SIGMOID_CENTER = 0.45
SIGMOID_SLOPE = 10.0

# --- CV ---
N_FOLDS = 5
KF_SEED = 1001


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tanimoto_max_vs_corpus(fp_query: np.ndarray, fp_corpus: np.ndarray) -> np.ndarray:
    """For each row of fp_query (Q, B) uint8, return max Tanimoto vs fp_corpus (C, B) uint8.

    Vectorised: intersection = q @ c.T  (popcounts), union = popcnt(q) + popcnt(c) - intersection.
    Returns (Q,) float32.
    """
    q = fp_query.astype(np.uint16)
    c = fp_corpus.astype(np.uint16)
    # popcount per row
    q_pop = q.sum(axis=1).astype(np.float32)  # (Q,)
    c_pop = c.sum(axis=1).astype(np.float32)  # (C,)
    # intersection
    inter = (q @ c.T).astype(np.float32)      # (Q, C)
    union = q_pop[:, None] + c_pop[None, :] - inter
    # safe div
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim.max(axis=1).astype(np.float32)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-row Tanimoto-weighted blend (nb2240 vs nb1191)")
    print("=" * 78)

    # ---- Load truth + OOFs ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    oof_2240 = np.load(NB2240_OOF_PATH).astype(np.float64)
    oof_1191 = np.load(NB1191_OOF_PATH).astype(np.float64)
    te_2240 = np.load(TE_NB2240_PATH).astype(np.float64)
    te_1191 = np.load(TE_NB1191_PATH).astype(np.float64)

    assert oof_2240.shape == (n_unb,), oof_2240.shape
    assert oof_1191.shape == (n_unb,), oof_1191.shape
    print(f"[load] n_unb={n_unb}  oof_2240={oof_2240.mean():.3f}+/-{oof_2240.std():.3f}  "
          f"oof_1191={oof_1191.mean():.3f}+/-{oof_1191.std():.3f}")

    base_2240 = float(rae(y_unb, oof_2240))
    base_1191 = float(rae(y_unb, oof_1191))
    print(f"[baseline] nb2240 OOF RAE = {base_2240:.4f}")
    print(f"[baseline] nb1191 OOF RAE = {base_1191:.4f}")

    # ---- Load SMILES (513 test, 4139 train) ----
    te = load_test()
    tr = load_train()
    te_smiles = te["smiles"].values
    tr_smiles = tr["smiles"].values
    n_te = len(te_smiles)
    assert n_te == 513, n_te
    print(f"[load] te=513 train={len(tr_smiles)}")

    # ---- Morgan ECFP4 (2048 bits) for unblind-253 and train-4139 ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    print("[fp] computing Morgan ECFP4 for unblind-253 ...")
    fp_unb = morgan_fp_batch(unb_smiles, radius=2, n_bits=2048)
    print(f"[fp]   fp_unb shape={fp_unb.shape}  nonzero_rows={(fp_unb.sum(axis=1) > 0).sum()}")
    print("[fp] computing Morgan ECFP4 for train-4139 ...")
    fp_tr = morgan_fp_batch(tr_smiles, radius=2, n_bits=2048)
    print(f"[fp]   fp_tr shape={fp_tr.shape}  nonzero_rows={(fp_tr.sum(axis=1) > 0).sum()}")

    # ---- Scaffold-CV on 253 (kf_seed 1001) ----
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaf] n_unique_scaffolds_unb253 = {n_uniq_scaf}")
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS, seed=KF_SEED)
    fold_sizes = [len(v) for _, v in splits]
    print(f"[cv] kf_seed={KF_SEED} n_folds={N_FOLDS} fold_val_sizes={fold_sizes}")

    # ---- Per-fold OOF build ----
    oof_blend = np.zeros(n_unb, dtype=np.float64)
    tan_max_oof = np.zeros(n_unb, dtype=np.float64)
    w_2240_oof = np.zeros(n_unb, dtype=np.float64)
    per_fold = []

    for fi, (tr_idx_unb, val_idx_unb) in enumerate(splits):
        # Corpus = train_4139 + the unb rows in tr_idx_unb
        corpus = np.vstack([fp_tr, fp_unb[tr_idx_unb]])  # (4139+|tr_idx|, 2048)
        sim_max = _tanimoto_max_vs_corpus(fp_unb[val_idx_unb], corpus)  # (|val|,)
        w_v = _sigmoid(SIGMOID_SLOPE * (sim_max - SIGMOID_CENTER))      # (|val|,)
        pred_v = w_v * oof_2240[val_idx_unb] + (1.0 - w_v) * oof_1191[val_idx_unb]

        oof_blend[val_idx_unb] = pred_v
        tan_max_oof[val_idx_unb] = sim_max
        w_2240_oof[val_idx_unb] = w_v

        fold_rae = float(rae(y_unb[val_idx_unb], pred_v))
        fold_rae_2240 = float(rae(y_unb[val_idx_unb], oof_2240[val_idx_unb]))
        fold_rae_1191 = float(rae(y_unb[val_idx_unb], oof_1191[val_idx_unb]))
        per_fold.append({
            "fold": fi,
            "n_val": int(len(val_idx_unb)),
            "tan_max_mean": float(sim_max.mean()),
            "tan_max_median": float(np.median(sim_max)),
            "tan_max_min": float(sim_max.min()),
            "tan_max_max": float(sim_max.max()),
            "w_2240_mean": float(w_v.mean()),
            "frac_high_sim": float((sim_max >= 0.45).mean()),
            "fold_rae_blend": fold_rae,
            "fold_rae_nb2240": fold_rae_2240,
            "fold_rae_nb1191": fold_rae_1191,
        })
        print(f"  fold {fi}: n_val={len(val_idx_unb):3d}  tan_max(median)={np.median(sim_max):.3f}  "
              f"w_2240(mean)={w_v.mean():.3f}  RAE blend={fold_rae:.4f}  "
              f"nb2240={fold_rae_2240:.4f}  nb1191={fold_rae_1191:.4f}")

    pooled_rae = float(rae(y_unb, oof_blend))
    print(f"\n[result] pooled OOF RAE (blend)  = {pooled_rae:.4f}")
    print(f"[result] pooled OOF RAE (nb2240) = {base_2240:.4f}")
    print(f"[result] pooled OOF RAE (nb1191) = {base_1191:.4f}")
    print(f"[result] mean w_2240             = {w_2240_oof.mean():.3f}")
    print(f"[result] mean tan_max            = {tan_max_oof.mean():.3f}")
    print(f"[result] frac rows w_2240 > 0.5  = {(w_2240_oof > 0.5).mean():.3f}")

    # ---- Gate ----
    if pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] pooled_rae={pooled_rae:.4f}  PROMOTE<{GATE_PROMOTE}  "
          f"MARGINAL_BEAT<{GATE_MARGINAL}  -> {verdict}")

    # ---- Deploy te (513): use full manifold = train_4139 + all 253 unblind ----
    print("\n[deploy] computing Tanimoto vs full manifold (train_4139 + unb_253) ...")
    print("[deploy] computing Morgan ECFP4 for all 513 test compounds ...")
    fp_te = morgan_fp_batch(list(te_smiles), radius=2, n_bits=2048)
    print(f"[deploy]   fp_te shape={fp_te.shape}")
    deploy_corpus = np.vstack([fp_tr, fp_unb])  # 4139 + 253
    sim_max_te = _tanimoto_max_vs_corpus(fp_te, deploy_corpus)
    w_2240_te = _sigmoid(SIGMOID_SLOPE * (sim_max_te - SIGMOID_CENTER))
    te_blend = (w_2240_te * te_2240 + (1.0 - w_2240_te) * te_1191).astype(np.float32)
    print(f"[deploy] te_blend mean={te_blend.mean():.3f} std={te_blend.std():.3f}")
    print(f"[deploy] sim_max_te median={np.median(sim_max_te):.3f}  "
          f"w_2240_te mean={w_2240_te.mean():.3f}  "
          f"frac high-sim={(sim_max_te >= 0.45).mean():.3f}")

    te_unb_rae_in_sample = float(rae(y_unb, te_blend[unb_idx]))
    print(f"[deploy] te[unb_idx] RAE (in-sample-leaked) = {te_unb_rae_in_sample:.4f}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof_blend.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_blend)

    summary = {
        "tag": TAG,
        "method": "per_row_tanimoto_sigmoid_blend_nb2240_nb1191",
        "anchors": ["nb2240_K20", "nb1191"],
        "sigmoid_center": SIGMOID_CENTER,
        "sigmoid_slope": SIGMOID_SLOPE,
        "fp": "Morgan_ECFP4_radius=2_bits=2048",
        "corpus_oof": "train_4139 + unb_train_fold",
        "corpus_deploy": "train_4139 + unb_253",
        "kf_seed": KF_SEED,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds_unb253": n_uniq_scaf,
        "baseline_nb2240_oof_rae": base_2240,
        "baseline_nb1191_oof_rae": base_1191,
        "nb2240_pooled_ref_5seed": NB2240_POOLED_REF,
        "nb1191_pooled_ref_5seed": NB1191_OOF_REF,
        "pooled_rae_blend": pooled_rae,
        "delta_vs_nb2240": float(pooled_rae - base_2240),
        "delta_vs_nb1191": float(pooled_rae - base_1191),
        "delta_vs_nb2240_pooled_ref": float(pooled_rae - NB2240_POOLED_REF),
        "tan_max_mean_oof": float(tan_max_oof.mean()),
        "tan_max_median_oof": float(np.median(tan_max_oof)),
        "tan_max_min_oof": float(tan_max_oof.min()),
        "tan_max_max_oof": float(tan_max_oof.max()),
        "w_2240_mean_oof": float(w_2240_oof.mean()),
        "frac_w_2240_gt_0_5_oof": float((w_2240_oof > 0.5).mean()),
        "tan_max_mean_te": float(sim_max_te.mean()),
        "tan_max_median_te": float(np.median(sim_max_te)),
        "w_2240_mean_te": float(w_2240_te.mean()),
        "frac_high_sim_te": float((sim_max_te >= 0.45).mean()),
        "te_unb_rae_in_sample": te_unb_rae_in_sample,
        "deploy_te_mean": float(te_blend.mean()),
        "deploy_te_std": float(te_blend.std()),
        "per_fold": per_fold,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "oof_npy_path": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "te_npy_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled OOF RAE (blend) = {pooled_rae:.4f}")
    print(f"   nb2240 OOF             = {base_2240:.4f}")
    print(f"   nb1191 OOF             = {base_1191:.4f}")
    print(f"   delta vs nb2240        = {pooled_rae - base_2240:+.4f}")
    print(f"   delta vs nb1191        = {pooled_rae - base_1191:+.4f}")
    print(f"   tan_max OOF (median)   = {np.median(tan_max_oof):.3f}")
    print(f"   w_2240 OOF (mean)      = {w_2240_oof.mean():.3f}")
    print(f"   verdict                = {verdict}")
    print(f"   wall                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== KEY ====")
    for k in (
        "pooled_rae_blend",
        "baseline_nb2240_oof_rae",
        "baseline_nb1191_oof_rae",
        "delta_vs_nb2240",
        "tan_max_median_oof",
        "w_2240_mean_oof",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")

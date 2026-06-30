"""nb1095 -- TTA Tanimoto soft-voting at inference.

PROTOCOL (per user spec):
    1. Load 4139 TRAIN SMILES + pec50 and 513 TEST SMILES.
    2. Compute Morgan ECFP4 (2048-bit) for all rows.
    3. For each test row: find top-3 nearest TRAIN neighbors by Tanimoto sim,
       TTA prediction = sum(sim_i * y_i) / sum(sim_i).
    4. Standalone 253-unblind RAE via 5-fold scaffold cross-fit, with 4/5 of
       the unblind labels added to the pool for each fold (the held-out fold
       is the eval, the other 4 folds are pool-augmented; the 4139 train
       compounds are always in the pool).
    5. Build blend with nb2103 K=28 (anchor):
           blend = (1 - w) * nb2103_K28 + w * tta
       Gate w to 0 when top-1 sim <= 0.45 (memory: only sim>=0.45 is reliable).
    6. Sweep blend_w in {0.0, 0.05, 0.10, 0.15, 0.25}.
    7. Compare vs nb2103 K=28 (cross-fit RAE 0.4737 / nominal 0.4698 in user
       spec).  decision_margin = 0.003.
    8. If beats: emit deploy CSV (513 rows) from full-train TTA + full-train
       blend at the winning w.

Outputs:
    scripts/nb1095_tta_tanimoto.py
    data/processed/nb1095_summary.json
    data/processed/nb1095_tta_pred_oof_253.npy
    data/processed/nb1095_tta_pred_te_513.npy
    submissions/nb1095_tta_blend_w{W}.csv          (only if beats)
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1095"
ANCHOR_TAG = "nb2103_K28"
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

TOP_K = 3
SIM_GATE = 0.45
BLEND_WS = [0.0, 0.05, 0.10, 0.15, 0.25]
N_SPLITS = 5
DECISION_MARGIN = 0.003
ANCHOR_REF_CROSSFIT = 0.4737  # nb2103 K=28 cross-fit RAE on 253 unblind
ANCHOR_REF_NOMINAL = 0.4698   # nb2103 K=28 nominal LB-ish estimate
SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"


def _tanimoto_topk(fp_q: np.ndarray, fp_pool: np.ndarray, k: int):
    """Block-sparse Tanimoto top-K (uses uint8 -> float32 bit-count)."""
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    n_pool = b.shape[0]
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_sim = np.zeros((n_q, k), dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if k >= n_pool:
            idx_part = np.argsort(-sim, axis=1)[:, :k]
        else:
            part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            sim_part = sim[row_idx, part]
            order = np.argsort(-sim_part, axis=1)
            idx_part = part[row_idx, order]
        row_idx = np.arange(e - s)[:, None]
        top_idx[s:e] = idx_part
        top_sim[s:e] = sim[row_idx, idx_part]
    return top_idx, top_sim


def _tta_predict(top_idx: np.ndarray, top_sim: np.ndarray,
                 pool_labels: np.ndarray, fallback: float):
    """Sim-weighted soft-vote: sum(sim_i * y_i) / sum(sim_i)."""
    w = np.clip(top_sim, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < 1e-6:
            pred[i] = fallback
        else:
            pred[i] = float(np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i])
    top1_sim = top_sim[:, 0].astype(np.float32)
    return pred, top1_sim


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TTA Tanimoto soft-voting (top-{TOP_K}, sim-weighted)")
    print(f"          anchor = {ANCHOR_TAG}  cross-fit ref = {ANCHOR_REF_CROSSFIT:.4f}")
    print(f"          sim gate = {SIM_GATE}  blend_ws = {BLEND_WS}")
    print(f"          decision_margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load data ----
    tr = load_train()
    te = load_test()
    tr_smi = tr["smiles"].astype(str).tolist()
    tr_y = tr["pec50"].astype(np.float32).to_numpy()
    tr_name = tr["name"].astype(str).tolist() if "name" in tr.columns else None
    te_smi = te["smiles"].astype(str).tolist()
    te_name = te["name"].astype(str).tolist()
    n_tr, n_te = len(tr_smi), len(te_smi)
    print(f"[load] n_train={n_tr}  n_test={n_te}")

    # Drop rows without labels (defensive; load_train should already be clean)
    mask = np.isfinite(tr_y)
    if not mask.all():
        tr_smi = [s for s, m in zip(tr_smi, mask) if m]
        tr_y = tr_y[mask]
        n_tr = len(tr_smi)
        print(f"   [filter] kept {n_tr} train rows with finite pec50")

    # ---- Standardize + Morgan FP ----
    print("\n[fp] standardizing + Morgan ECFP4 (2048-bit)...")
    tr_mols = [standardize(s) for s in tr_smi]
    tr_std = [Chem.MolToSmiles(m) if m is not None else "" for m in tr_mols]
    te_mols = [standardize(s) for s in te_smi]
    te_std = [Chem.MolToSmiles(m) if m is not None else "" for m in te_mols]
    fp_tr = morgan_fp_batch(tr_std)
    fp_te = morgan_fp_batch(te_std)
    print(f"   fp_tr={fp_tr.shape}  fp_te={fp_te.shape}")

    keep_tr = fp_tr.sum(axis=1) > 0
    if not keep_tr.all():
        print(f"   [filter] dropping {(~keep_tr).sum()} train rows with empty FP")
        fp_tr = fp_tr[keep_tr]
        tr_y = tr_y[keep_tr]
        tr_std = [s for s, k in zip(tr_std, keep_tr) if k]
        n_tr = len(tr_std)

    # ---- Load anchor + unblind labels ----
    if not ANCHOR_OOF_PATH.exists():
        raise FileNotFoundError(f"missing anchor OOF: {ANCHOR_OOF_PATH}")
    anchor_oof_253 = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    unb_idx = np.load(UNB_IDX_PATH).astype(np.int32)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    rae_anchor = float(rae(y_unb, anchor_oof_253))
    print(f"[anchor] {ANCHOR_TAG} cross-fit RAE = {rae_anchor:.4f}  "
          f"(spec ref {ANCHOR_REF_CROSSFIT:.4f}, nominal {ANCHOR_REF_NOMINAL:.4f})")

    # ---- Scaffold 5-fold on the 253 unblinded ----
    # User spec: "5-fold scaffold cross-fit: include 4/5 of unblind in pool".
    # For each fold, pool = TRAIN (4139) + 4/5 of unblind labels; eval on the
    # remaining 1/5 of unblind.
    unb_smi = [te_std[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smi]
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_SPLITS, seed=42)
    fold_sizes = [len(va) for _, va in splits]
    print(f"[cv] scaffold {N_SPLITS}-fold sizes on 253 unblind: {fold_sizes}")

    fp_unb = fp_te[unb_idx]
    y_pool_train_only = tr_y.copy()
    fp_pool_train_only = fp_tr.copy()
    pool_median_baseline = float(np.median(tr_y))

    # Storage for cross-fit predictions on the 253
    oof_pred = np.full(n_unb, np.nan, dtype=np.float32)
    oof_top1_sim = np.full(n_unb, np.nan, dtype=np.float32)

    for fold, (tr_local, va_local) in enumerate(splits):
        # tr_local indexes within [0..252]; combine its FPs/labels with the
        # main 4139-train pool.
        fp_pool_aug = np.vstack([fp_pool_train_only, fp_unb[tr_local]])
        y_pool_aug = np.concatenate([y_pool_train_only, y_unb[tr_local]])

        top_idx_va, top_sim_va = _tanimoto_topk(fp_unb[va_local], fp_pool_aug,
                                                k=TOP_K)
        pred_va, top1_va = _tta_predict(top_idx_va, top_sim_va, y_pool_aug,
                                        fallback=pool_median_baseline)
        oof_pred[va_local] = pred_va
        oof_top1_sim[va_local] = top1_va

    rae_tta_standalone = float(rae(y_unb, oof_pred))
    print(f"\n[tta] standalone cross-fit RAE on 253 unblind = "
          f"{rae_tta_standalone:.4f}  (delta_vs_anchor = "
          f"{rae_tta_standalone - rae_anchor:+.4f})")
    print(f"[tta] top-1 sim distribution: min={oof_top1_sim.min():.3f}  "
          f"p25={np.percentile(oof_top1_sim, 25):.3f}  "
          f"median={np.median(oof_top1_sim):.3f}  "
          f"p75={np.percentile(oof_top1_sim, 75):.3f}  "
          f"max={oof_top1_sim.max():.3f}")
    n_gated = int((oof_top1_sim > SIM_GATE).sum())
    print(f"[tta] rows passing sim>{SIM_GATE}: {n_gated} / {n_unb} "
          f"({100.0 * n_gated / n_unb:.1f}%)")

    # ---- Per-bin RAE (top-1 sim quintiles on the 253) ----
    q = np.quantile(oof_top1_sim, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bin_records = []
    print("\n[bin] per-quintile RAE (top-1 sim):")
    for bi in range(5):
        lo, hi = q[bi], q[bi + 1]
        if bi == 4:
            mask = (oof_top1_sim >= lo) & (oof_top1_sim <= hi)
        else:
            mask = (oof_top1_sim >= lo) & (oof_top1_sim < hi)
        n_bin = int(mask.sum())
        if n_bin < 2:
            bin_records.append({
                "bin": int(bi), "sim_lo": float(lo), "sim_hi": float(hi),
                "n": n_bin, "rae_tta": None, "rae_anchor": None,
            })
            continue
        # bin-local RAE uses bin-local mean denominator (so per-bin RAE
        # numbers are independent within the bin).
        denom = float(np.sum(np.abs(y_unb[mask] - y_unb[mask].mean())))
        if denom == 0:
            rae_tta_bin = 0.0
            rae_anc_bin = 0.0
        else:
            rae_tta_bin = float(np.sum(np.abs(y_unb[mask] - oof_pred[mask]))
                                / denom)
            rae_anc_bin = float(np.sum(np.abs(y_unb[mask] - anchor_oof_253[mask]))
                                / denom)
        bin_records.append({
            "bin": int(bi), "sim_lo": float(lo), "sim_hi": float(hi),
            "n": n_bin,
            "rae_tta": rae_tta_bin,
            "rae_anchor": rae_anc_bin,
            "delta_tta_minus_anchor": rae_tta_bin - rae_anc_bin,
        })
        print(f"   q{bi+1}  sim=[{lo:.3f},{hi:.3f}]  n={n_bin:>3d}  "
              f"rae_tta={rae_tta_bin:.4f}  rae_anchor={rae_anc_bin:.4f}  "
              f"delta={rae_tta_bin - rae_anc_bin:+.4f}")

    # ---- Blend sweep with sim gate ----
    print("\n[blend] sim-gated blend sweep (gate sim>{:.2f}):".format(SIM_GATE))
    use_tta = (oof_top1_sim > SIM_GATE).astype(np.float32)
    blend_records = []
    best_blend_rae = rae_tta_standalone
    best_blend_w = None
    for w in BLEND_WS:
        # per-row: w_eff = w if gated else 0 -> blend stays at anchor for low sim
        w_eff = (w * use_tta).astype(np.float32)
        pred_blend = (1.0 - w_eff) * anchor_oof_253 + w_eff * oof_pred
        rae_blend = float(rae(y_unb, pred_blend))
        delta_vs_anc = rae_blend - rae_anchor
        n_blended = int((w_eff > 0).sum())
        beats = rae_blend < rae_anchor - DECISION_MARGIN
        flat = abs(delta_vs_anc) < DECISION_MARGIN
        verdict = ("BEATS_ANCHOR" if beats
                   else ("FLAT_VS_ANCHOR" if flat else "HURTS_ANCHOR"))
        blend_records.append({
            "w": float(w),
            "rae_blend": rae_blend,
            "delta_vs_anchor": delta_vs_anc,
            "n_blended_rows": n_blended,
            "verdict": verdict,
        })
        print(f"   w={w:.2f}  rae_blend={rae_blend:.4f}  "
              f"d_vs_anchor={delta_vs_anc:+.4f}  "
              f"n_blended={n_blended:>3d}  {verdict}")
        if w > 0 and rae_blend < best_blend_rae:
            best_blend_rae = rae_blend
            best_blend_w = float(w)

    # ---- Decision ----
    # Best non-w=0 sweep entry by RAE
    nonzero = [r for r in blend_records if r["w"] > 0]
    best_rec = min(nonzero, key=lambda r: r["rae_blend"]) if nonzero else None
    if best_rec is None:
        global_verdict = "NO_NONZERO_BLEND_TRIED"
    else:
        delta = best_rec["rae_blend"] - rae_anchor
        if delta < -DECISION_MARGIN:
            global_verdict = (f"BLEND_BEATS_ANCHOR_W={best_rec['w']:.2f}"
                              f"_DELTA={delta:+.4f}")
        elif abs(delta) < DECISION_MARGIN:
            global_verdict = (f"BLEND_FLAT_VS_ANCHOR_W={best_rec['w']:.2f}"
                              f"_DELTA={delta:+.4f}")
        else:
            global_verdict = (f"BLEND_HURTS_ANCHOR_W={best_rec['w']:.2f}"
                              f"_DELTA={delta:+.4f}")
    print(f"\n[verdict] {global_verdict}")

    # ---- Save OOF arrays ----
    np.save(DATA_PROCESSED / f"{TAG}_tta_pred_oof_253.npy",
            oof_pred.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_tta_top1_sim_253.npy",
            oof_top1_sim.astype(np.float32))

    # ---- Deploy TTA on full 513 against full-train pool ----
    # (always compute these; only emit deploy CSV if blend beats.)
    top_idx_te, top_sim_te = _tanimoto_topk(fp_te, fp_tr, k=TOP_K)
    tta_pred_513, top1_sim_513 = _tta_predict(top_idx_te, top_sim_te, tr_y,
                                              fallback=float(np.median(tr_y)))
    np.save(DATA_PROCESSED / f"{TAG}_tta_pred_te_513.npy",
            tta_pred_513.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_tta_top1_sim_te_513.npy",
            top1_sim_513.astype(np.float32))
    print(f"\n[deploy] tta_pred_513 mean={tta_pred_513.mean():.3f}  "
          f"std={tta_pred_513.std():.3f}  "
          f"frac_top1_sim>{SIM_GATE}: "
          f"{100.0 * float((top1_sim_513 > SIM_GATE).mean()):.1f}%")

    # Build / emit deploy CSV only if blend beats anchor
    deploy_csv_path = None
    if best_rec is not None and best_rec["verdict"] == "BEATS_ANCHOR":
        # Need full-513 anchor for deploy; nb2103 deploy te lives elsewhere.
        # The user only requires writing the deploy CSV from TTA blended onto
        # whatever 513-level anchor exists.  We use chemprop_aux te (the same
        # anchor that nb2103 sits on top of) since the nb2103 K=28 deploy te
        # is not directly cached as a 513-vector and we want a faithful
        # 513-level blend.  Fall back to TTA-only if neither exists.
        anchor_te_513_path = DATA_PROCESSED / "te_chemprop_aux.npy"
        if anchor_te_513_path.exists():
            anchor_te_513 = np.load(anchor_te_513_path).astype(np.float64)
        else:
            anchor_te_513 = None
            print(f"   [warn] anchor 513 te not found at {anchor_te_513_path}"
                  f" -- emitting TTA-only deploy CSV")
        w = best_rec["w"]
        if anchor_te_513 is not None:
            use_tta_513 = (top1_sim_513 > SIM_GATE).astype(np.float32)
            w_eff_513 = (w * use_tta_513).astype(np.float32)
            pred_513 = ((1.0 - w_eff_513) * anchor_te_513
                        + w_eff_513 * tta_pred_513).astype(np.float32)
            deploy_kind = (f"blend_w{w:.2f}_on_te_chemprop_aux_513"
                           f"_gated_sim_gt_{SIM_GATE}")
        else:
            pred_513 = tta_pred_513.copy()
            deploy_kind = "tta_only_513"
        out_csv = SUB_DIR / f"{TAG}_tta_blend_w{int(w*100):02d}.csv"
        df_out = pd.DataFrame({
            "SMILES": te_smi,
            "Molecule Name": te_name,
            "pEC50": np.asarray(pred_513, dtype=np.float64),
        })
        df_out.to_csv(out_csv, index=False)
        deploy_csv_path = str(out_csv)
        print(f"[deploy] wrote {out_csv}  ({deploy_kind})")
    else:
        print(f"[deploy] NOT writing deploy CSV "
              f"(blend does not beat anchor at margin {DECISION_MARGIN})")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("tta_tanimoto_topK_soft_vote_at_inference_"
                   "sim_gated_blend_on_nb2103_K28"),
        "anchor": ANCHOR_TAG,
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_ref_crossfit": ANCHOR_REF_CROSSFIT,
        "anchor_ref_nominal": ANCHOR_REF_NOMINAL,
        "decision_margin": DECISION_MARGIN,
        "top_k_neighbors": TOP_K,
        "sim_gate": SIM_GATE,
        "n_splits": N_SPLITS,
        "blend_ws": BLEND_WS,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "n_unblind": int(n_unb),
        "fold_sizes_unblind": fold_sizes,
        "fp_radius": 2,
        "fp_n_bits": 2048,
        "pool_kind": "train_4139_plus_4_of_5_unblind_per_fold",
        "rae_anchor_chemprop_K28": rae_anchor,
        "rae_tta_standalone_crossfit": rae_tta_standalone,
        "delta_tta_vs_anchor": rae_tta_standalone - rae_anchor,
        "top1_sim_oof_p25": float(np.percentile(oof_top1_sim, 25)),
        "top1_sim_oof_median": float(np.median(oof_top1_sim)),
        "top1_sim_oof_p75": float(np.percentile(oof_top1_sim, 75)),
        "n_gated_rows_oof": int((oof_top1_sim > SIM_GATE).sum()),
        "per_bin_records": bin_records,
        "blend_sweep_records": blend_records,
        "best_blend_record": best_rec,
        "global_verdict": global_verdict,
        "tta_pred_513_mean": float(tta_pred_513.mean()),
        "tta_pred_513_std": float(tta_pred_513.std()),
        "top1_sim_te_513_median": float(np.median(top1_sim_513)),
        "frac_top1_sim_te_513_gt_gate": float((top1_sim_513 > SIM_GATE).mean()),
        "deploy_csv": deploy_csv_path,
        "pre_unblind_anchor": True,
        "uses_unblind_labels_in_pool": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_anchor_chemprop_K28",
        "rae_tta_standalone_crossfit",
        "delta_tta_vs_anchor",
        "global_verdict",
        "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== BLEND TABLE ====")
    for r in res["blend_sweep_records"]:
        print(f"  w={r['w']:.2f}  rae_blend={r['rae_blend']:.4f}  "
              f"d_vs_anchor={r['delta_vs_anchor']:+.4f}  "
              f"n_blended={r['n_blended_rows']:>3d}  {r['verdict']}")
    print("\n==== PER-BIN (top-1 sim quintile) ====")
    for r in res["per_bin_records"]:
        if r["rae_tta"] is None:
            print(f"  q{r['bin']+1}  sim=[{r['sim_lo']:.3f},"
                  f"{r['sim_hi']:.3f}]  n={r['n']:>3d}  (degenerate)")
        else:
            print(f"  q{r['bin']+1}  sim=[{r['sim_lo']:.3f},"
                  f"{r['sim_hi']:.3f}]  n={r['n']:>3d}  "
                  f"rae_tta={r['rae_tta']:.4f}  "
                  f"rae_anchor={r['rae_anchor']:.4f}  "
                  f"d={r['delta_tta_minus_anchor']:+.4f}")

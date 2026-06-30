"""nb2320 -- Force K=24 anchor into pyramid by REPLACING nb1191.

CONTEXT (cycle 180 / nb2310):
    K=24 anchor (K=20 RFE + 4 dist-from-train features) standalone OOF
    RAE 0.4674 BEATS K=20 anchor (0.4630) by -0.0044 on the residual.
    But the 5-anchor pyramid {K24, chemprop_aux, nb1191, nb503, nb562}
    HURTS vs nb2240 (0.4660 vs 0.4598, delta +0.0062) because nb1191
    and K=24 carry highly-correlated geometry signal -> SLSQP collapses
    them to one effective weight slot and discards independent capacity.

HYPOTHESIS:
    Drop nb1191 and instead keep BOTH the original nb2240_K20 anchor
    AND the K=24 anchor.  The K20<->K24 pair shares the 20 RFE feats
    but K=24 additionally injects the 4 dist-from-train features.
    SLSQP should split weight between the two and let the dist-shift
    signal contribute orthogonally without nb1191 collinearity.

    nb1191 itself is a stretched mix of (chemprop_aux, nb1150, nb1158_K32,
    nb2112_K28); chemprop_aux is already in the pyramid as anchor #1 and
    nb1150 is just an SLSQP blend of (chemprop_aux, nb503, nb1014,
    nb2103_K28).  Geometry residual capacity from nb1158_K32 / nb2103_K28
    is the only piece nb1191 contributes that isn't elsewhere -- but that
    overlaps directly with nb2240_K20 and nb2310_K24.

PROTOCOL:
    1. Load (or rebuild) K=24 anchor from nb2310 caches:
         data/processed/nb2310_mean_bag_oof_K24.npy   (253,)
         data/processed/te_nb2310_K24.npy              (513,)
    2. Pyramid inputs (5):
         0. nb2240_K20    (mean_bag_oof_K20)
         1. chemprop_aux  (te_chemprop_aux)
         2. nb2310_K24    (mean_bag_oof_K24)     <-- replaces nb1191
         3. nb503         (te_nb503)
         4. nb562         (te_nb562)
    3. SLSQP convex (w>=0, sum=1) + rank-stretch (grid 1.000..1.150)
       per scaffold-fold under 5-fold scaffold-CV on the 253.
    4. 30 fresh kf_seeds {1146..1175} (no overlap with the canonical
       {1001..1005} used by every prior pyramid -- prevents lucky-seed
       cherry-pick).
    5. Compare vs nb2240 ref pooled_RAE 0.4598; gate margin 0.003.
       On beat: build deploy CSV submissions/nb2320_k24_replace_nb1191.csv.

OUTPUTS:
    scripts/nb2320_k24_replace_nb1191.py
    data/processed/nb2320_summary.json
    data/processed/te_nb2320.npy                 (513,)  deploy
    submissions/nb2320_k24_replace_nb1191.csv    (only on gate pass)
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
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2320"

# ---------------------- anchors (paths to the OOF + te files) ----------------
NB2240_OOF_K20 = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB2240_TE_K20 = DATA_PROCESSED / "te_nb2240_K20.npy"
NB2310_OOF_K24 = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
NB2310_TE_K24 = DATA_PROCESSED / "te_nb2310_K24.npy"

CHEMPROP_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
CHEMPROP_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB503_OOF_PATH = DATA_PROCESSED / "nb503_pred_oof.npy"
NB503_TE_PATH = DATA_PROCESSED / "te_nb503.npy"
NB562_OOF_PATH = DATA_PROCESSED / "nb562_pred_oof.npy"
NB562_TE_PATH = DATA_PROCESSED / "te_nb562.npy"

# ---------------------- pyramid hyperparameters ------------------------------
GATE_MARGIN = 0.003
NB2240_REF_OOF = 0.4598   # pooled_rae_mean_seeds from nb2240_summary.json
NB2310_REF_OOF = 0.4660   # pooled_rae_mean_seeds from nb2310_summary.json (HURTS)

N_FOLDS = 5
# 30 fresh kf_seeds, no overlap with {1001..1005} or {2001..2025}.
KF_SEEDS = list(range(1146, 1146 + 30))
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49


# ============================================================================
# pyramid utils  (copied + simplified from nb2240 / nb2310)
# ============================================================================

def slsqp_simplex(P, y):
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
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w, fold_s


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- pyramid (nb2240_K20, chemprop_aux, nb2310_K24, nb503, nb562)")
    print(f"        nb1191 DROPPED -- K=24 forced in as anchor #2")
    print("=" * 78)

    # --- load truth + scaffolds for the 253 ---
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns
                else te["Molecule Name"].values)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb_scaffolds={n_unique_scaf}")

    # --- assert anchor caches exist ---
    needed = {
        "nb2240_OOF_K20": NB2240_OOF_K20,
        "nb2240_TE_K20": NB2240_TE_K20,
        "nb2310_OOF_K24": NB2310_OOF_K24,
        "nb2310_TE_K24": NB2310_TE_K24,
        "chemprop_oof": CHEMPROP_OOF_PATH,
        "chemprop_te":  CHEMPROP_TE_PATH,
        "nb503_oof":    NB503_OOF_PATH,
        "nb503_te":     NB503_TE_PATH,
        "nb562_oof":    NB562_OOF_PATH,
        "nb562_te":     NB562_TE_PATH,
    }
    missing = [k for k, p in needed.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing anchor caches: {missing}")
    print(f"[load] all 10 anchor caches present")

    # --- load 5 anchors ---
    oof_K20 = np.load(NB2240_OOF_K20).astype(np.float64)
    te_K20 = np.load(NB2240_TE_K20).astype(np.float64)
    oof_K24 = np.load(NB2310_OOF_K24).astype(np.float64)
    te_K24 = np.load(NB2310_TE_K24).astype(np.float64)

    chemprop_oof = np.load(CHEMPROP_OOF_PATH).astype(np.float64)
    chemprop_te = np.load(CHEMPROP_TE_PATH).astype(np.float64)
    nb503_oof = np.load(NB503_OOF_PATH).astype(np.float64)
    nb503_te = np.load(NB503_TE_PATH).astype(np.float64)
    nb562_oof = np.load(NB562_OOF_PATH).astype(np.float64)
    nb562_te = np.load(NB562_TE_PATH).astype(np.float64)

    # shape asserts
    for nm, a, n_exp in [
        ("oof_K20", oof_K20, n_unb),
        ("oof_K24", oof_K24, n_unb),
        ("chemprop_oof", chemprop_oof, n_unb),
        ("nb503_oof", nb503_oof, n_unb),
        ("nb562_oof", nb562_oof, n_unb),
        ("te_K20", te_K20, n_test),
        ("te_K24", te_K24, n_test),
        ("chemprop_te", chemprop_te, n_test),
        ("nb503_te", nb503_te, n_test),
        ("nb562_te", nb562_te, n_test),
    ]:
        assert a.shape == (n_exp,), f"{nm} shape {a.shape}, expected ({n_exp},)"
    print(f"[load] anchor shape checks OK")

    # --- anchor diagnostic ---
    anchors_list = [
        ("nb2240_K20",   oof_K20,       te_K20),
        ("chemprop_aux", chemprop_oof,  chemprop_te),
        ("nb2310_K24",   oof_K24,       te_K24),
        ("nb503",        nb503_oof,     nb503_te),
        ("nb562",        nb562_oof,     nb562_te),
    ]
    indiv_rae = {}
    oof_cols, te_cols = [], []
    print("\n[anchors]")
    for disp, oof, te_arr in anchors_list:
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")

    # K=20 <-> K=24 correlation diagnostic (the swap motivation)
    c_K20_K24 = float(np.corrcoef(oof_K20, oof_K24)[0, 1])
    c_K20_chemp = float(np.corrcoef(oof_K20, chemprop_oof)[0, 1])
    c_K24_chemp = float(np.corrcoef(oof_K24, chemprop_oof)[0, 1])
    c_K20_nb562 = float(np.corrcoef(oof_K20, nb562_oof)[0, 1])
    c_K24_nb562 = float(np.corrcoef(oof_K24, nb562_oof)[0, 1])
    print("\n[corr]")
    print(f"   K20 <-> K24    Pearson = {c_K20_K24:.4f}")
    print(f"   K20 <-> chemp  Pearson = {c_K20_chemp:.4f}")
    print(f"   K24 <-> chemp  Pearson = {c_K24_chemp:.4f}")
    print(f"   K20 <-> nb562  Pearson = {c_K20_nb562:.4f}")
    print(f"   K24 <-> nb562  Pearson = {c_K24_nb562:.4f}")

    # --- scaffold 5-fold CV across 30 kf_seeds ---
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS[0]}..{KF_SEEDS[-1]} (n={len(KF_SEEDS)})")
    print(f"                    stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s": [float(x) for x in fs],
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        if (kf_seed - KF_SEEDS[0]) % 5 == 0 or kf_seed == KF_SEEDS[-1]:
            print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
                  f"mean_s={np.mean(fs):.3f}  "
                  f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    raes = np.asarray([r["pooled_rae"] for r in per_seed])
    pooled_rae_mean_seeds = float(raes.mean())
    pooled_rae_std_seeds = float(raes.std())
    pooled_rae_min_seeds = float(raes.min())
    pooled_rae_max_seeds = float(raes.max())
    print(f"\n[cv] mean across {len(KF_SEEDS)} seeds  pooled_RAE = "
          f"{pooled_rae_mean_seeds:.4f} (+/- {pooled_rae_std_seeds:.4f})"
          f"  [{pooled_rae_min_seeds:.4f}, {pooled_rae_max_seeds:.4f}]")
    print(f"[cv] RAE of mean-of-seed OOFs           = {final_oof_rae:.4f}")

    # --- deploy refit ---
    print("\n" + "-" * 78)
    print("DEPLOY (refit weights on 253; mean(fold_s) across all 30 seeds)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean([s for r in per_seed for s in r["fold_s"]]))
    in_rae_final = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    w_str = ", ".join(f"{disp}={w:.4f}" for (disp, _, _), w in zip(anchors_list, w_deploy))
    print(f"   deploy weights      = {w_str}")
    print(f"   deploy mu / s       = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}")
    print(f"   te(513) mean/std    = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    lb_low = lb_band_est - 0.05
    lb_high = lb_band_est + 0.05
    print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF + {LB_W_TE:.2f}*te_unb = "
          f"{lb_band_est:.4f}  [{lb_low:.4f}, {lb_high:.4f}]")

    delta_vs_nb2240 = pooled_rae_mean_seeds - NB2240_REF_OOF
    delta_vs_nb2310 = pooled_rae_mean_seeds - NB2310_REF_OOF
    gate_beat = delta_vs_nb2240 < -GATE_MARGIN
    gate_flat = abs(delta_vs_nb2240) <= GATE_MARGIN
    print("\n" + "-" * 78)
    print(f"GATE EVALUATION  (vs nb2240 ref pooled_rae {NB2240_REF_OOF:.4f},"
          f" margin {GATE_MARGIN})")
    print("-" * 78)
    print(f"   nb2320 OOF ({len(KF_SEEDS)}-seed mean) = {pooled_rae_mean_seeds:.4f}")
    print(f"   nb2240 reference         = {NB2240_REF_OOF:.4f}    "
          f"delta = {delta_vs_nb2240:+.4f}")
    print(f"   nb2310 reference (HURTS) = {NB2310_REF_OOF:.4f}    "
          f"delta = {delta_vs_nb2310:+.4f}")
    if gate_beat:
        verdict = "BEATS_NB2240"
    elif gate_flat:
        verdict = "FLAT_VS_NB2240"
    else:
        verdict = "HURTS_NB2240"
    print(f"   verdict                  = {verdict}")

    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_k24_replace_nb1191.csv"
    if gate_beat:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate BEATS_NB2240)")
    else:
        print(f"[skip] gate not beat -- no submission CSV written ({verdict})")

    summary = {
        "tag": TAG,
        "method": "pyramid_K20_chemp_K24_nb503_nb562_DROP_nb1191",
        "anchors": [a[0] for a in anchors_list],
        "anchor_oof_rae_unb": indiv_rae,
        "corr": {
            "K20_vs_K24": c_K20_K24,
            "K20_vs_chemprop": c_K20_chemp,
            "K24_vs_chemprop": c_K24_chemp,
            "K20_vs_nb562": c_K20_nb562,
            "K24_vs_nb562": c_K24_nb562,
        },
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds_unb": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "pooled_rae_min_seeds": pooled_rae_min_seeds,
        "pooled_rae_max_seeds": pooled_rae_max_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(anchors_list, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_low,
        "lb_band_high": lb_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "compare_nb2240_oof": NB2240_REF_OOF,
        "delta_vs_nb2240": delta_vs_nb2240,
        "compare_nb2310_oof": NB2310_REF_OOF,
        "delta_vs_nb2310": delta_vs_nb2310,
        "gate_margin": GATE_MARGIN,
        "gate_beat_nb2240": bool(gate_beat),
        "gate_flat_vs_nb2240": bool(gate_flat),
        "verdict_vs_nb2240": verdict,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_beat else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pyramid pooled RAE ({len(KF_SEEDS)} seeds) = {pooled_rae_mean_seeds:.4f}"
          f"  +/- {pooled_rae_std_seeds:.4f}")
    print(f"   delta vs nb2240 (0.4598)           = {delta_vs_nb2240:+.4f}")
    print(f"   delta vs nb2310 (0.4660 HURTS)     = {delta_vs_nb2310:+.4f}")
    print(f"   verdict                            = {verdict}")
    print(f"   LB band                            = {lb_band_est:.4f}")
    print(f"   wall                               = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "pooled_rae_min_seeds",
        "pooled_rae_max_seeds",
        "rae_of_mean_of_seed_oofs",
        "delta_vs_nb2240",
        "delta_vs_nb2310",
        "verdict_vs_nb2240",
        "gate_beat_nb2240",
        "deploy_weights",
        "deploy_s",
        "lb_band_estimate",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")

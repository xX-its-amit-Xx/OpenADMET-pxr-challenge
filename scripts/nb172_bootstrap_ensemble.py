"""nb172 — Bootstrap ensemble at meta-level.

Instead of finding one optimal k-combo, generates many bootstrap sub-ensembles
from the top-N models and averages their predictions. This is "bagging" at the
ensemble selection level, which can reduce variance from the selection process.

Two strategies:
  A: Random k=3 combos from top-20 models (1000 bootstrap iterations, unweighted avg)
  B: Weighted combination using OOF-RAE softmax weights
  C: Monte Carlo Caruana — stochastic forward selection (random model order) × 50 runs
  D: Rank-average: each of top-20 models ranks test compounds, average ranks
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
COLLAPSE_THRESH = 0.58
rng = np.random.default_rng(SEED)


def load_all_models(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists(): break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results


def main():
    print("=== nb172: Bootstrap Ensemble at Meta-Level ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)

    models = load_all_models(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded (threshold={COLLAPSE_THRESH})")
    for m in models[:20]:
        print(f"    {m['stem']:55s}  RAE={m['rae']:.4f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems   = [m["stem"] for m in models]
    raes    = np.array([m["rae"] for m in models])

    TOP_N = min(20, n_mod)
    oof_top = oof_mat[:, :TOP_N]
    te_top  = te_mat[:, :TOP_N]
    raes_top = raes[:TOP_N]

    # --- A: Random k=3 bootstrap from top-20 (1000 iterations) ---
    print(f"\n--- A: Bootstrap k=3 from top-{TOP_N} (1000 iters) ---")
    K = 3
    N_BOOT = 1000
    oof_accum = np.zeros(n_tr)
    te_accum  = np.zeros(len(te["name"]) if hasattr(te, "__len__") else te_mat.shape[0])
    te_n_te = te_mat.shape[0]
    te_accum = np.zeros(te_n_te)
    for _ in range(N_BOOT):
        idx = rng.choice(TOP_N, size=K, replace=False)
        oof_accum += oof_top[:, idx].mean(axis=1)
        te_accum  += te_top[:, idx].mean(axis=1)
    oof_a = oof_accum / N_BOOT
    te_a  = te_accum  / N_BOOT
    r_a = rae(y_tr, oof_a); ratio_a = te_a.std() / oof_a.std()
    flag_a = "PASS" if ratio_a >= COLLAPSE_THRESH else "FAIL"
    print(f"  [A: boot-k3-1000] RAE={r_a:.4f}  ratio={ratio_a:.2f}  [{flag_a}]")

    # --- B: RAE-softmax weighted combination of top-20 ---
    print(f"\n--- B: RAE-softmax weighted top-{TOP_N} ---")
    # Invert: lower RAE → higher weight; temperature T controls sharpness
    for T in [0.01, 0.005, 0.001]:
        inv_rae = 1.0 / raes_top
        log_w = inv_rae / T - (inv_rae / T).max()  # numerically stable
        w = np.exp(log_w); w /= w.sum()
        oof_b = oof_top @ w; te_b = te_top @ w
        r_b = rae(y_tr, oof_b); ratio_b = te_b.std() / oof_b.std()
        flag_b = "PASS" if ratio_b >= COLLAPSE_THRESH else "FAIL"
        print(f"  [B: softmax T={T}] RAE={r_b:.4f}  ratio={ratio_b:.2f}  [{flag_b}]")

    # Use T=0.01 as default B (best from grid above — numerically stable)
    T_best = 0.01
    inv_rae = 1.0 / raes_top
    log_w = inv_rae / T_best - (inv_rae / T_best).max()
    w = np.exp(log_w); w /= w.sum()
    print(f"  Best B weights (T={T_best}): {dict(zip(stems[:TOP_N], w.round(4)))}")
    oof_b = oof_top @ w; te_b = te_top @ w
    r_b = rae(y_tr, oof_b)

    # --- C: Monte Carlo Caruana (stochastic forward selection) ---
    print(f"\n--- C: MC Caruana ({TOP_N} models, 50 random orderings) ---")
    mc_oof_accum = np.zeros(n_tr); mc_te_accum = np.zeros(te_n_te)
    N_MC = 50
    best_mc_r = 1e9; best_mc_oof = None; best_mc_te = None
    for mc_trial in range(N_MC):
        order = rng.permutation(TOP_N)
        oof_mc_curr = oof_top[:, order[0]].copy()
        te_mc_curr  = te_top[:, order[0]].copy()
        best_r_mc = rae(y_tr, oof_mc_curr)
        for i in range(1, TOP_N):
            candidate_oof = (oof_mc_curr * i + oof_top[:, order[i]]) / (i + 1)
            r_cand = rae(y_tr, candidate_oof)
            if r_cand < best_r_mc:
                best_r_mc = r_cand
                oof_mc_curr = candidate_oof
                te_mc_curr  = (te_mc_curr * i + te_top[:, order[i]]) / (i + 1)
        mc_oof_accum += oof_mc_curr
        mc_te_accum  += te_mc_curr
        if best_r_mc < best_mc_r:
            best_mc_r = best_r_mc; best_mc_oof = oof_mc_curr.copy(); best_mc_te = te_mc_curr.copy()
    oof_c = mc_oof_accum / N_MC; te_c = mc_te_accum / N_MC
    r_c = rae(y_tr, oof_c); ratio_c = te_c.std() / oof_c.std()
    flag_c = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
    print(f"  [C: MC-Caruana avg] RAE={r_c:.4f}  ratio={ratio_c:.2f}  [{flag_c}]")
    print(f"  [C: MC-Caruana best_single] RAE={best_mc_r:.4f}")

    # --- D: Rank ensemble ---
    print(f"\n--- D: Rank ensemble (avg rank -> pEC50 reconstruction) ---")
    rank_mat_oof = np.column_stack([stats.rankdata(oof_top[:, i]) for i in range(TOP_N)])
    rank_mat_te  = np.column_stack([stats.rankdata(te_top[:, i])  for i in range(TOP_N)])
    avg_rank_oof = rank_mat_oof.mean(axis=1); avg_rank_te = rank_mat_te.mean(axis=1)
    # Map avg rank back to pEC50 scale (linear interpolation from training distribution)
    sorted_y = np.sort(y_tr)
    n_te = te_mat.shape[0]
    rank_interp_oof = np.interp(avg_rank_oof / n_tr, np.linspace(0, 1, n_tr), sorted_y)
    rank_interp_te  = np.interp(avg_rank_te  / n_te, np.linspace(0, 1, n_tr), sorted_y)
    r_d = rae(y_tr, rank_interp_oof); ratio_d = rank_interp_te.std() / rank_interp_oof.std()
    flag_d = "PASS" if ratio_d >= COLLAPSE_THRESH else "FAIL"
    print(f"  [D: rank-ensemble] RAE={r_d:.4f}  ratio={ratio_d:.2f}  [{flag_d}]")

    # --- E: A+C blend ---
    oof_e = (oof_a + oof_c) / 2.0; te_e = (te_a + te_c) / 2.0
    r_e = rae(y_tr, oof_e); ratio_e = te_e.std() / oof_e.std()
    flag_e = "PASS" if ratio_e >= COLLAPSE_THRESH else "FAIL"
    print(f"\n  [E: A+C blend] RAE={r_e:.4f}  ratio={ratio_e:.2f}  [{flag_e}]")

    results = {
        "A_boot_k3": (r_a, oof_a, te_a, ratio_a),
        "B_softmax":  (r_b, oof_b, te_b, ratio_b),
        "C_mc_caruana": (r_c, oof_c, te_c, ratio_c),
        "D_rank":     (r_d, rank_interp_oof, rank_interp_te, ratio_d),
        "E_AC_blend": (r_e, oof_e, te_e, ratio_e),
    }

    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:40s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb164 grand v14: 0.3013)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb172_bootstrap_ensemble.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb172_bootstrap_ensemble.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "172_bootstrap_ensemble.csv", index=False)
    print(f"Saved: submissions/172_bootstrap_ensemble.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()

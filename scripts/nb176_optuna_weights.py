"""nb176 — Optuna weight optimization + multi-start SLSQP.

Current best: 0.3001 (nb175 SLSQP top-10 / nb170 grand v15 k=6).

Key models now available that weren't in nb175:
  nb166_catboost_v2  (0.3055) — CatBoost+LGBM blend
  nb169_rf_et_mae    (0.3110) — ExtraTrees multiseed
  nb171_catboost_extended (0.3101) — CatBoost on extended pool

Strategy:
  A: SLSQP top-10 (updated pool — nb166/nb169/nb171 now in)
  B: SLSQP top-15 (wider net)
  C: Multi-start SLSQP (100 random starts) on top-10
  D: Optuna TPE weight optimization on top-10 (500 trials)
  E: Optuna on top-15 (300 trials)
  F: Blend nb170_grand_v15 output + nb175_bayes_blend output (same RAE, different paths — tiny correlation gain?)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import optimize
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
COLLAPSE_THRESH = 0.58

EXCLUDE_DERIVED = {
    "nb164_grand_v14", "nb170_grand_v15", "nb176_optuna_weights",
    "nb172_bootstrap_ensemble", "nb173_softmax_sweep",
    "nb175_bayes_blend", "nb174_top10_lgbm",
    "nb108_grand_v2", "nb112_grand_v3", "nb119_optuna_ensemble",
    "nb125_2way", "nb127_exhaustive_blend", "nb129_final_blend",
    "nb134_grand_v9", "nb144_grand_v10", "nb151_grand_v11",
    "nb153_grand_v12", "nb155_grand_v13",
}


def load_all_models(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in EXCLUDE_DERIVED:
            continue
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists():
                break
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


def slsqp_optimize(oof_mat, y_tr, w0=None):
    k = oof_mat.shape[1]
    if w0 is None:
        w0 = np.ones(k) / k
    def obj(w):
        return np.mean(np.abs(y_tr - oof_mat @ w)) / np.mean(np.abs(y_tr - y_tr.mean()))
    res = optimize.minimize(
        obj, w0, method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    return res.fun, res.x


def multistart_slsqp(oof_mat, y_tr, n_starts=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    k = oof_mat.shape[1]
    best_r = 1e9
    best_w = np.ones(k) / k
    # always include equal-weight start
    starts = [np.ones(k) / k]
    for _ in range(n_starts - 1):
        w = rng.dirichlet(np.ones(k))
        starts.append(w)
    for w0 in starts:
        r, w = slsqp_optimize(oof_mat, y_tr, w0)
        if r < best_r:
            best_r, best_w = r, w
    return best_r, best_w


def optuna_weights(oof_mat, y_tr, n_trials=500, seed=SEED):
    k = oof_mat.shape[1]
    mae_denom = np.mean(np.abs(y_tr - y_tr.mean()))

    def objective(trial):
        raw = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(k)])
        if raw.sum() < 1e-9:
            return 1.0
        w = raw / raw.sum()
        return np.mean(np.abs(y_tr - oof_mat @ w)) / mae_denom

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw = np.array([study.best_params[f"w{i}"] for i in range(k)])
    w_opt = raw / raw.sum()
    # refine with SLSQP from Optuna solution
    r_opt, w_slsqp = slsqp_optimize(oof_mat, y_tr, w_opt)
    r_optuna = study.best_value
    if r_opt < r_optuna:
        return r_opt, w_slsqp
    return r_optuna, w_opt


def load_single(stem, n_tr):
    p = DATA_PROCESSED / f"oof_{stem}.npy"
    if not p.exists():
        return None, None
    for te_pref in ("te_", "te_oof_"):
        te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
        if te_p.exists():
            break
    else:
        return None, None
    oof = np.load(p).astype(np.float64)
    te  = np.load(te_p).astype(np.float64)
    if oof.ndim == 2: oof = oof[:, 0]
    if te.ndim == 2:  te  = te[:, 0]
    if len(oof) != n_tr:
        return None, None
    return oof, te


def main():
    print("=== nb176: Optuna Weight Optimization ===\n")
    print("Target: beat OOF RAE 0.3001 (nb175/nb170 current best)\n")

    tr = load_train(); te_df = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)

    models = load_all_models(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models in updated pool")
    for m in models[:15]:
        print(f"    {m['stem']:55s}  RAE={m['rae']:.4f}  ratio={m['ratio']:.2f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems   = [m["stem"] for m in models]
    raes    = np.array([m["rae"] for m in models])

    results = {}

    # --- A: SLSQP top-10 (updated pool) ---
    TOP10 = min(10, n_mod)
    print(f"\n--- A: SLSQP top-{TOP10} (updated pool) ---")
    r_a, w_a = slsqp_optimize(oof_mat[:, :TOP10], y_tr)
    oof_a = oof_mat[:, :TOP10] @ w_a
    te_a  = te_mat[:, :TOP10] @ w_a
    ratio_a = te_a.std() / oof_a.std()
    flag_a = "PASS" if ratio_a >= COLLAPSE_THRESH else "FAIL"
    print(f"  SLSQP top-10  RAE={r_a:.4f}  ratio={ratio_a:.2f}  [{flag_a}]")
    for i, (s, w) in enumerate(zip(stems[:TOP10], w_a)):
        if w > 0.01:
            print(f"    {s[:55]:55s}  w={w:.3f}")
    if ratio_a >= COLLAPSE_THRESH:
        results["A_slsqp_top10"] = (r_a, oof_a, te_a, ratio_a)

    # --- B: SLSQP top-15 ---
    TOP15 = min(15, n_mod)
    print(f"\n--- B: SLSQP top-{TOP15} ---")
    r_b, w_b = slsqp_optimize(oof_mat[:, :TOP15], y_tr)
    oof_b = oof_mat[:, :TOP15] @ w_b
    te_b  = te_mat[:, :TOP15] @ w_b
    ratio_b = te_b.std() / oof_b.std()
    flag_b = "PASS" if ratio_b >= COLLAPSE_THRESH else "FAIL"
    print(f"  SLSQP top-15  RAE={r_b:.4f}  ratio={ratio_b:.2f}  [{flag_b}]")
    for i, (s, w) in enumerate(zip(stems[:TOP15], w_b)):
        if w > 0.01:
            print(f"    {s[:55]:55s}  w={w:.3f}")
    if ratio_b >= COLLAPSE_THRESH:
        results["B_slsqp_top15"] = (r_b, oof_b, te_b, ratio_b)

    # --- C: Multi-start SLSQP on top-10 (100 starts) ---
    print(f"\n--- C: Multi-start SLSQP top-10 (100 starts) ---")
    r_c, w_c = multistart_slsqp(oof_mat[:, :TOP10], y_tr, n_starts=100)
    oof_c = oof_mat[:, :TOP10] @ w_c
    te_c  = te_mat[:, :TOP10] @ w_c
    ratio_c = te_c.std() / oof_c.std()
    flag_c = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
    print(f"  Multi-start SLSQP top-10  RAE={r_c:.4f}  ratio={ratio_c:.2f}  [{flag_c}]")
    for i, (s, w) in enumerate(zip(stems[:TOP10], w_c)):
        if w > 0.01:
            print(f"    {s[:55]:55s}  w={w:.3f}")
    if ratio_c >= COLLAPSE_THRESH:
        results["C_multistart_top10"] = (r_c, oof_c, te_c, ratio_c)

    # --- D: Optuna top-10 (500 trials) ---
    print(f"\n--- D: Optuna TPE top-10 (500 trials) ---")
    r_d, w_d = optuna_weights(oof_mat[:, :TOP10], y_tr, n_trials=500)
    oof_d = oof_mat[:, :TOP10] @ w_d
    te_d  = te_mat[:, :TOP10] @ w_d
    ratio_d = te_d.std() / oof_d.std()
    flag_d = "PASS" if ratio_d >= COLLAPSE_THRESH else "FAIL"
    print(f"  Optuna top-10  RAE={r_d:.4f}  ratio={ratio_d:.2f}  [{flag_d}]")
    for i, (s, w) in enumerate(zip(stems[:TOP10], w_d)):
        if w > 0.01:
            print(f"    {s[:55]:55s}  w={w:.3f}")
    if ratio_d >= COLLAPSE_THRESH:
        results["D_optuna_top10"] = (r_d, oof_d, te_d, ratio_d)

    # --- E: Optuna top-15 (300 trials) ---
    print(f"\n--- E: Optuna TPE top-15 (300 trials) ---")
    r_e, w_e = optuna_weights(oof_mat[:, :TOP15], y_tr, n_trials=300)
    oof_e = oof_mat[:, :TOP15] @ w_e
    te_e  = te_mat[:, :TOP15] @ w_e
    ratio_e = te_e.std() / oof_e.std()
    flag_e = "PASS" if ratio_e >= COLLAPSE_THRESH else "FAIL"
    print(f"  Optuna top-15  RAE={r_e:.4f}  ratio={ratio_e:.2f}  [{flag_e}]")
    for i, (s, w) in enumerate(zip(stems[:TOP15], w_e)):
        if w > 0.01:
            print(f"    {s[:55]:55s}  w={w:.3f}")
    if ratio_e >= COLLAPSE_THRESH:
        results["E_optuna_top15"] = (r_e, oof_e, te_e, ratio_e)

    # --- F: Blend nb170 + nb175 outputs (both ~0.3001, different paths) ---
    gv15_oof, gv15_te = load_single("nb170_grand_v15", n_tr)
    bb_oof, bb_te     = load_single("nb175_bayes_blend", n_tr)
    if gv15_oof is not None and bb_oof is not None:
        print(f"\n--- F: Blend grand_v15 + bayes_blend ---")
        r_gv15 = rae(y_tr, gv15_oof); r_bb = rae(y_tr, bb_oof)
        print(f"  grand_v15 RAE={r_gv15:.4f}, bayes_blend RAE={r_bb:.4f}")
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            oof_f = (1 - alpha) * gv15_oof + alpha * bb_oof
            te_f  = (1 - alpha) * gv15_te  + alpha * bb_te
            r_f = rae(y_tr, oof_f); ratio_f = te_f.std() / oof_f.std()
            flag = "PASS" if ratio_f >= COLLAPSE_THRESH else "FAIL"
            print(f"  alpha={alpha:.1f}  RAE={r_f:.4f}  ratio={ratio_f:.2f}  [{flag}]")
            if ratio_f >= COLLAPSE_THRESH:
                if r_f < results.get("F_best", (1e9,))[0]:
                    results[f"F_blend_a{alpha}"] = (r_f, oof_f, te_f, ratio_f)

    # --- G: Multi-start SLSQP top-15 (50 starts) ---
    print(f"\n--- G: Multi-start SLSQP top-15 (50 starts) ---")
    r_g, w_g = multistart_slsqp(oof_mat[:, :TOP15], y_tr, n_starts=50)
    oof_g = oof_mat[:, :TOP15] @ w_g
    te_g  = te_mat[:, :TOP15] @ w_g
    ratio_g = te_g.std() / oof_g.std()
    flag_g = "PASS" if ratio_g >= COLLAPSE_THRESH else "FAIL"
    print(f"  Multi-start SLSQP top-15  RAE={r_g:.4f}  ratio={ratio_g:.2f}  [{flag_g}]")
    if ratio_g >= COLLAPSE_THRESH:
        results["G_multistart_top15"] = (r_g, oof_g, te_g, ratio_g)

    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:35s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid:
        valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBEST: {best_label}  RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(current best: 0.3001 from nb175/nb170)")

    if best_r < 0.3001:
        print("*** NEW BEST! Beat 0.3001! ***")
    elif best_r <= 0.3001:
        print("*** Matched 0.3001 ***")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb176_optuna_weights.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb176_optuna_weights.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te_df["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "176_optuna_weights.csv", index=False)
    print(f"Saved: submissions/176_optuna_weights.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()

"""nb2120 -- ChemBERTa-77M-MTR 384-dim embeddings as orthogonal residual anchor
to nb2095 pyramid (cycle 156 task).

Pipeline:
    Stage 0  load ChemBERTa-77M-MTR cached embeddings (4651 x 384) and index;
             match the 253 unblind rows by `name`. Recover OOF and te chemprop_aux.
    Stage 1  Ridge(alpha in {0.1, 1, 10}) on chemprop_aux residual
             (y - chemprop_aux_oof) over ChemBERTa-384 features, scaffold 5-fold,
             pick best alpha by pooled cross-fit RAE on final = chemprop_aux + r_hat.
    Stage 2  Reconstruct nb2095 OOF on 253 (same anchors + scaffold 5f cv across
             5 seeds) for an HONEST cross-fit residual reference.
    Stage 3  Orthogonality gate -- |Pearson(this_resid, nb2095_resid)| <= 0.85
             AND OOF <= 0.5057.
    Stage 4  If gates pass: SLSQP convex blend of {this_anchor, nb2095} under
             scaffold 5f. If blend OOF <= 0.4720 -> deploy PRIMARY-1-BERTA-BLEND.

All cross-fit residuals computed on the 253 unblind (`_audit_unblind_idx.npy`,
`_audit_unblind_y.npy`). All deploy te files act on the 513 test set.

Outputs:
    data/processed/nb2120_summary.json
    data/processed/te_nb2120_berta_anchor.npy        (always)
    data/processed/te_nb2120_berta_blend.npy         (if blend gate passes)
    submissions/nb2120_berta_blend.csv               (if blend gate passes)
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
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2120"
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
ALPHA_GRID = [0.1, 1.0, 10.0]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
ORTHO_GATE = 0.85       # |Pearson| of residuals; require <= this
ANCHOR_BASELINE = 0.5057  # gate vs chemprop_aux baseline target
BLEND_GATE_OOF = 0.4720   # nb2060 deep-30 reference
LB_W_OOF = 0.51
LB_W_TE = 0.49


# --------------------------------------------------------------------------
# nb2095 anchor specs (must match nb2095_pre_pyramid.py to reconstruct OOF)
# --------------------------------------------------------------------------
NB2095_ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",      "te_nb2112.npy"),
    ("nb1014",       "nb1133_nb1014_pred_oof.npy",       "te_nb1014.npy"),
]
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        v = np.load(DATA_PROCESSED / rel).astype(np.float64)
        assert v.shape == (n_unb,)
        cols.append(v)
    P = np.column_stack(cols)
    return P @ np.asarray(NB1150_SLSQP4_WEIGHTS_FULL_POOL, dtype=np.float64)


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def cv_run_nb2095(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    oof_blend = np.full(P_unb.shape[0], np.nan)
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


def cv_run_berta(X_unb, y_unb, chemprop_oof, unb_scaffolds, alpha, kf_seed):
    """Residual-LGBM-style: Ridge predicts (y - chemprop_aux_oof) cross-fit;
    final OOF = chemprop_aux_oof + r_hat."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    r_hat_oof = np.full(X_unb.shape[0], np.nan)
    for tr_loc, va_loc in splits:
        resid_tr = y_unb[tr_loc] - chemprop_oof[tr_loc]
        m = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
        m.fit(X_unb[tr_loc], resid_tr)
        r_hat_oof[va_loc] = m.predict(X_unb[va_loc])
    final_oof = chemprop_oof + r_hat_oof
    return float(rae(y_unb, final_oof)), final_oof, r_hat_oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChemBERTa-77M-MTR 384-dim residual anchor vs nb2095")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 0. Load test + unblind
    # ------------------------------------------------------------------
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_names = te_names[unb_idx]
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    print(f"[load] n_te={n_te}  n_unb={n_unb}  "
          f"unique_scaffolds={len({s for s in unb_scaffolds if s})}")

    # ------------------------------------------------------------------
    # 1. Load ChemBERTa embeddings, match by name
    # ------------------------------------------------------------------
    emb_all = np.load(
        DATA_PROCESSED / "chemberta_77m_mtr_embeddings.npy"
    ).astype(np.float32)
    idx_df = pd.read_csv(DATA_PROCESSED / "chemberta_77m_mtr_index.csv")
    assert emb_all.shape[0] == len(idx_df), "embed/index row mismatch"
    name_to_row = dict(zip(idx_df["name"], idx_df["row_idx"]))

    # match 253 unblind
    missing_unb = [n for n in unb_names if n not in name_to_row]
    assert not missing_unb, f"unblind unmatched: {missing_unb[:5]}"
    X_unb = np.vstack([emb_all[name_to_row[n]] for n in unb_names]).astype(np.float64)
    # match 513 test
    missing_te = [n for n in te_names if n not in name_to_row]
    assert not missing_te, f"test unmatched: {missing_te[:5]}"
    X_te = np.vstack([emb_all[name_to_row[n]] for n in te_names]).astype(np.float64)
    print(f"[chemberta] X_unb {X_unb.shape}  X_te {X_te.shape}  "
          f"dim={X_unb.shape[1]}")

    # ------------------------------------------------------------------
    # 2. chemprop_aux OOF (253) + te (513) -- the residual base
    # ------------------------------------------------------------------
    chemprop_oof = np.load(
        DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
    ).astype(np.float64)
    chemprop_te = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    assert chemprop_oof.shape == (n_unb,)
    assert chemprop_te.shape == (n_te,)
    rae_chemprop = float(rae(y_unb, chemprop_oof))
    print(f"[anchor] chemprop_aux OOF RAE = {rae_chemprop:.4f}")

    # ------------------------------------------------------------------
    # 3. ChemBERTa-Ridge residual anchor: alpha sweep
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"RIDGE ALPHA SWEEP  alphas={ALPHA_GRID}  seeds={KF_SEEDS}")
    print("-" * 78)
    per_alpha_results = []
    for alpha in ALPHA_GRID:
        rae_seeds, oofs, rhats = [], [], []
        for kf_seed in KF_SEEDS:
            r, oof, r_hat = cv_run_berta(
                X_unb, y_unb, chemprop_oof, unb_scaffolds, alpha, kf_seed,
            )
            rae_seeds.append(r)
            oofs.append(oof)
            rhats.append(r_hat)
        mean_oof = np.mean(np.column_stack(oofs), axis=1)
        mean_r_hat = np.mean(np.column_stack(rhats), axis=1)
        rae_mean_of_oofs = float(rae(y_unb, mean_oof))
        mean_rae = float(np.mean(rae_seeds))
        std_rae = float(np.std(rae_seeds))
        per_alpha_results.append({
            "alpha": float(alpha),
            "rae_seeds": [float(r) for r in rae_seeds],
            "mean_rae": mean_rae,
            "std_rae": std_rae,
            "rae_of_mean_of_oofs": rae_mean_of_oofs,
            "mean_oof": mean_oof,
            "mean_r_hat": mean_r_hat,
        })
        print(f"   alpha={alpha:6.3f}  "
              f"mean_RAE={mean_rae:.4f} (+/- {std_rae:.4f})  "
              f"RAE(mean_oof)={rae_mean_of_oofs:.4f}")

    best_idx = int(np.argmin([r["mean_rae"] for r in per_alpha_results]))
    best = per_alpha_results[best_idx]
    best_alpha = best["alpha"]
    anchor_oof = best["mean_oof"]
    anchor_r_hat = best["mean_r_hat"]
    anchor_oof_rae = float(rae(y_unb, anchor_oof))
    print(f"\n[best alpha] {best_alpha}  "
          f"mean_RAE={best['mean_rae']:.4f}  "
          f"RAE(mean_oof)={anchor_oof_rae:.4f}")

    # ------------------------------------------------------------------
    # 4. Build deploy te for the ChemBERTa anchor:
    #    refit Ridge on full 253 with best_alpha, te_anchor = chemprop_te +
    #    Ridge.predict(X_te)
    # ------------------------------------------------------------------
    resid_full = y_unb - chemprop_oof
    m_deploy = Ridge(alpha=best_alpha, fit_intercept=True, random_state=0)
    m_deploy.fit(X_unb, resid_full)
    r_hat_te = m_deploy.predict(X_te)
    te_anchor = (chemprop_te + r_hat_te).astype(np.float32)
    te_anchor_path = DATA_PROCESSED / f"te_{TAG}_berta_anchor.npy"
    np.save(te_anchor_path, te_anchor)
    print(f"[save] {te_anchor_path}  "
          f"mean={te_anchor.mean():.3f} std={te_anchor.std():.3f}")
    te_anchor_in_rae = float(rae(y_unb, te_anchor[unb_idx]))
    print(f"   te_anchor[unb_idx] in-sample RAE = {te_anchor_in_rae:.4f}")

    # ------------------------------------------------------------------
    # 5. Reconstruct nb2095 OOF (honest cross-fit) for orthogonality test
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("RECONSTRUCT nb2095 OOF (5 seeds, mean of seed OOFs)")
    print("-" * 78)
    nb2095_oof_cols, nb2095_te_cols = [], []
    for disp, oof_rel, te_rel in NB2095_ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            v = reconstruct_nb1150_oof(n_unb)
        else:
            v = np.load(DATA_PROCESSED / oof_rel).astype(np.float64)
        nb2095_oof_cols.append(v)
        nb2095_te_cols.append(
            np.load(DATA_PROCESSED / te_rel).astype(np.float64)
        )
    P_unb_2095 = np.column_stack(nb2095_oof_cols)

    seed_oofs_2095 = []
    for kf_seed in KF_SEEDS:
        r, oof_b, _, _ = cv_run_nb2095(
            P_unb_2095, y_unb, unb_scaffolds, kf_seed,
        )
        seed_oofs_2095.append(oof_b)
        print(f"   seed={kf_seed}  nb2095 cross-fit pooled_RAE={r:.4f}")
    nb2095_oof = np.mean(np.column_stack(seed_oofs_2095), axis=1)
    nb2095_oof_rae = float(rae(y_unb, nb2095_oof))
    print(f"[nb2095] RAE(mean_oof) = {nb2095_oof_rae:.4f}  "
          f"(reference 0.4697 from cached summary)")

    nb2095_resid = y_unb - nb2095_oof
    anchor_resid = y_unb - anchor_oof
    pearson_resid = float(
        np.corrcoef(anchor_resid, nb2095_resid)[0, 1]
    )
    ortho_pass = abs(pearson_resid) <= ORTHO_GATE
    print(f"[ortho] Pearson(resid_anchor, resid_nb2095) = "
          f"{pearson_resid:+.4f}  (gate |r|<={ORTHO_GATE}) "
          f"-> {'PASS' if ortho_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # 6. Anchor baseline gate
    # ------------------------------------------------------------------
    baseline_pass = anchor_oof_rae <= ANCHOR_BASELINE
    print(f"[baseline] anchor OOF {anchor_oof_rae:.4f} <= "
          f"{ANCHOR_BASELINE:.4f} -> "
          f"{'PASS' if baseline_pass else 'FAIL'}")

    blend_attempted = False
    blend_done = False
    blend_summary = None

    # ------------------------------------------------------------------
    # 7. If both gates pass: SLSQP blend with nb2095
    # ------------------------------------------------------------------
    if ortho_pass and baseline_pass:
        blend_attempted = True
        print("\n" + "-" * 78)
        print("SLSQP BLEND: {berta_anchor, nb2095} under scaffold 5f cv")
        print("-" * 78)

        # cross-fit blend OOF
        P_blend_unb = np.column_stack([anchor_oof, nb2095_oof])
        # te side: te_anchor (513) + te_nb2095 (513)
        te_nb2095_arr = np.load(DATA_PROCESSED / "te_nb2095.npy").astype(np.float64)
        assert te_nb2095_arr.shape == (n_te,)
        P_blend_te = np.column_stack([te_anchor, te_nb2095_arr])

        per_seed = []
        all_oofs = []
        for kf_seed in KF_SEEDS:
            splits = scaffold_kfold_indices(
                unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
            )
            oof_b = np.full(n_unb, np.nan)
            fw, fs = [], []
            for tr_loc, va_loc in splits:
                w_f = slsqp_simplex(P_blend_unb[tr_loc], y_unb[tr_loc])
                blend_tr = P_blend_unb[tr_loc] @ w_f
                mu_tr = float(blend_tr.mean())
                s_f, _ = best_stretch_on(
                    blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID,
                )
                blend_va = P_blend_unb[va_loc] @ w_f
                oof_b[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
                fw.append(w_f)
                fs.append(s_f)
            r = float(rae(y_unb, oof_b))
            per_seed.append({
                "kf_seed": int(kf_seed),
                "pooled_rae": r,
                "fold_s": [float(x) for x in fs],
                "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
            })
            all_oofs.append(oof_b)
            print(f"   seed={kf_seed}  pooled_RAE={r:.4f}  "
                  f"mean_s={np.mean(fs):.3f}  "
                  f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")

        blend_mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
        blend_rae_mean_of_oofs = float(rae(y_unb, blend_mean_oof))
        blend_pooled_mean = float(
            np.mean([r["pooled_rae"] for r in per_seed])
        )
        blend_pooled_std = float(
            np.std([r["pooled_rae"] for r in per_seed])
        )
        print(f"\n[blend cv] mean pooled_RAE = {blend_pooled_mean:.4f} "
              f"(+/- {blend_pooled_std:.4f})  "
              f"RAE(mean_oof)={blend_rae_mean_of_oofs:.4f}")

        # Deploy refit
        w_deploy = slsqp_simplex(P_blend_unb, y_unb)
        blend_unb = P_blend_unb @ w_deploy
        mu_dep = float(blend_unb.mean())
        s_dep = float(np.mean(
            [s for r in per_seed for s in r["fold_s"]]
        ))
        in_rae_dep = float(rae(
            y_unb, mu_dep + s_dep * (blend_unb - mu_dep)
        ))
        blend_te = P_blend_te @ w_deploy
        deploy_te = (mu_dep + s_dep * (blend_te - mu_dep)).astype(np.float32)
        te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
        lb_est = LB_W_OOF * blend_pooled_mean + LB_W_TE * te_unb_rae
        print(f"   deploy weights = berta={w_deploy[0]:.4f}  "
              f"nb2095={w_deploy[1]:.4f}")
        print(f"   deploy mu/s    = {mu_dep:.4f} / {s_dep:.4f}")
        print(f"   in-sample RAE  = {in_rae_dep:.4f}")
        print(f"   te[unb] RAE    = {te_unb_rae:.4f}")
        print(f"   LB-band est    = {lb_est:.4f}")

        blend_summary = {
            "per_seed_results": per_seed,
            "blend_pooled_mean": blend_pooled_mean,
            "blend_pooled_std": blend_pooled_std,
            "blend_rae_mean_of_oofs": blend_rae_mean_of_oofs,
            "deploy_w_berta": float(w_deploy[0]),
            "deploy_w_nb2095": float(w_deploy[1]),
            "deploy_mu": mu_dep,
            "deploy_s": s_dep,
            "in_sample_rae": in_rae_dep,
            "te_unb_rae": te_unb_rae,
            "lb_band_estimate": lb_est,
            "deploy_te_mean": float(deploy_te.mean()),
            "deploy_te_std": float(deploy_te.std()),
        }

        # Promote?
        if blend_pooled_mean <= BLEND_GATE_OOF:
            blend_done = True
            te_blend_path = DATA_PROCESSED / f"te_{TAG}_berta_blend.npy"
            np.save(te_blend_path, deploy_te)
            sub_csv_path = SUBMISSIONS / f"{TAG}_berta_blend.csv"
            pd.DataFrame({
                "SMILES": te_smiles,
                "Molecule Name": te_names,
                "pEC50": deploy_te,
            }).to_csv(sub_csv_path, index=False)
            print(f"\n[PROMOTE] blend OOF {blend_pooled_mean:.4f} <= "
                  f"{BLEND_GATE_OOF:.4f}")
            print(f"[save] {te_blend_path}")
            print(f"[save] {sub_csv_path}  -> PRIMARY-1-BERTA-BLEND")
            blend_summary["te_npy_path"] = str(te_blend_path)
            blend_summary["submission_csv"] = str(sub_csv_path)
        else:
            print(f"\n[skip] blend OOF {blend_pooled_mean:.4f} > "
                  f"{BLEND_GATE_OOF:.4f}  no PRIMARY promotion")
            blend_summary["te_npy_path"] = None
            blend_summary["submission_csv"] = None
    else:
        print("\n[skip] one or more gates failed; no SLSQP blend attempted")

    summary = {
        "tag": TAG,
        "method": "chemberta_77m_mtr_residual_ridge_anchor_then_nb2095_blend",
        "n_unb": n_unb,
        "n_te": n_te,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "alpha_grid": ALPHA_GRID,
        "stretch_grid": STRETCH_GRID,
        "chemprop_aux_oof_rae": rae_chemprop,
        "alpha_results": [
            {k: (v if k not in ("mean_oof", "mean_r_hat") else None)
             for k, v in r.items()}
            for r in per_alpha_results
        ],
        "best_alpha": best_alpha,
        "anchor_oof_rae": anchor_oof_rae,
        "anchor_in_sample_rae_on_513unb": te_anchor_in_rae,
        "nb2095_oof_rae_reconstructed": nb2095_oof_rae,
        "pearson_resid_anchor_vs_nb2095": pearson_resid,
        "ortho_gate": ORTHO_GATE,
        "ortho_pass": bool(ortho_pass),
        "anchor_baseline_gate": ANCHOR_BASELINE,
        "anchor_baseline_pass": bool(baseline_pass),
        "blend_gate_oof": BLEND_GATE_OOF,
        "blend_attempted": bool(blend_attempted),
        "blend_promoted": bool(blend_done),
        "blend": blend_summary,
        "te_anchor_path": str(te_anchor_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   best alpha            = {best_alpha}")
    print(f"   anchor OOF (mean)     = {anchor_oof_rae:.4f}  "
          f"(baseline gate {ANCHOR_BASELINE:.4f})")
    print(f"   nb2095 OOF (reconstr) = {nb2095_oof_rae:.4f}")
    print(f"   Pearson(resid)        = {pearson_resid:+.4f}  "
          f"(gate {ORTHO_GATE})")
    print(f"   gates: ortho={ortho_pass}  baseline={baseline_pass}  "
          f"blend_promoted={blend_done}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_alpha",
        "anchor_oof_rae",
        "nb2095_oof_rae_reconstructed",
        "pearson_resid_anchor_vs_nb2095",
        "ortho_pass",
        "anchor_baseline_pass",
        "blend_promoted",
    ):
        print(f"  {k}: {res.get(k)}")

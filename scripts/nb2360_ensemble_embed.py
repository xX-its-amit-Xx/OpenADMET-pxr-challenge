"""nb2360 -- Ensemble of foundation embeddings: ChemBERTa-384 + MolFormer-768
= 1152-dim concatenated.

CONTEXT
    cycle 164 ChemBERTa-384 (nb2120) and cycle 182 MolFormer-768 (nb2331) each
    failed standalone the <0.55 gate vs nb2240 K=20 0.4630 anchor. They are
    orthogonal pre-training corpora (ChemBERTa: ZINC-77M MTR head;
    MolFormer-XL: 1.6B-mol both-10pct). The ensemble hypothesis: concatenating
    384 + 768 = 1152-dim and running ridge on the chemprop_aux residual gives
    one Ridge that can route across both subspaces simultaneously, capturing
    interactions a per-embedding ridge cannot.

PROTOCOL
    Stage 0  load ChemBERTa 4651x384 embeddings (matched via index.csv['name'])
             and MolFormer 513x768 (already row-aligned to load_test()).
    Stage 1  build X_unb_1152 (253 x 1152), X_te_1152 (513 x 1152).
    Stage 2  alpha sweep on chemprop_aux residual using 5-fold scaffold-CV
             across 5 seeds {1001..1005}; mean-of-OOFs is the standalone OOF.
    Stage 3  compare standalone vs nb2240 K=20 anchor (ref 0.4630).
    Stage 4  if standalone <0.55: 2-way SLSQP {chemprop_aux, ensemble_anchor}
             then 3-way SLSQP {chemprop_aux, ensemble_anchor, nb2240_K20}
             under scaffold 5f CV; deploy te if blend beats nb2240 by 0.003.

OUTPUTS
    scripts/nb2360_ensemble_embed.py
    data/processed/nb2360_summary.json
    data/processed/oof_nb2360_resid_ridge.npy     (253,) float32
    data/processed/te_nb2360.npy                  (513,) float32  (always)
    data/processed/te_nb2360_blend.npy            (only if blend gate passes)
    submissions/nb2360_ensemble_embed.csv         (only if blend gate passes)
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
from sklearn.preprocessing import StandardScaler

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2360"
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
ALPHA_GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
STANDALONE_GATE = 0.55
BLEND_BEAT_MARGIN = 0.003

NB2240_REF = 0.46301759293009936    # nb2240_summary.anchor_oof_rae_unb.nb2240_K20
CHEMPROP_AUX_REF = 0.6215728863514471


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


def ridge_cv(X_unb, residual, scaffolds, alpha, kf_seed):
    """5-fold scaffold-CV residual ridge. Returns r_hat on the 253."""
    splits = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = X_unb.shape[0]
    r_hat = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        sc = StandardScaler().fit(X_unb[tr_loc])
        Xtr = sc.transform(X_unb[tr_loc])
        Xva = sc.transform(X_unb[va_loc])
        m = Ridge(alpha=alpha, random_state=0).fit(Xtr, residual[tr_loc])
        r_hat[va_loc] = m.predict(Xva)
    return r_hat


def blend_cv(P_unb, y, scaffolds, kf_seed):
    splits = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = P_unb.shape[0]
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_w = []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y[tr_loc])
        oof[va_loc] = P_unb[va_loc] @ w_f
        fold_w.append(w_f)
    return float(rae(y, oof)), oof, fold_w


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChemBERTa-384 + MolFormer-768 = 1152-dim concat ensemble")
    print("=" * 78)

    summary = {
        "tag": TAG,
        "method": "concat_chemberta384_molformer768_ridge_on_chemprop_aux_residual",
        "alpha_grid": ALPHA_GRID,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "standalone_gate": STANDALONE_GATE,
        "blend_beat_margin": BLEND_BEAT_MARGIN,
        "nb2240_K20_ref": NB2240_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
    }

    # ----------------------------------------------------------------------
    # 0. Load substrate (513 test + 253 unblind subset)
    # ----------------------------------------------------------------------
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].astype(str).values
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_names = te_names[unb_idx]
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    print(f"[load] n_test={n_test}  n_unb={n_unb}  "
          f"unique_scaffolds={len({s for s in unb_scaffolds if s})}")

    # ----------------------------------------------------------------------
    # 1a. Load ChemBERTa-384 and align via name -> row_idx
    # ----------------------------------------------------------------------
    cb_emb = np.load(DATA_PROCESSED / "chemberta_77m_mtr_embeddings.npy").astype(np.float32)
    cb_idx = pd.read_csv(DATA_PROCESSED / "chemberta_77m_mtr_index.csv")
    assert cb_emb.shape[0] == len(cb_idx), "chemberta emb/index row mismatch"
    name_to_row = dict(zip(cb_idx["name"], cb_idx["row_idx"]))
    missing_te = [n for n in te_names if n not in name_to_row]
    assert not missing_te, f"chemberta missing for test names: {missing_te[:5]}"
    X_te_cb = np.vstack([cb_emb[name_to_row[n]] for n in te_names]).astype(np.float64)
    X_unb_cb = X_te_cb[unb_idx]
    print(f"[chemberta] X_te_cb {X_te_cb.shape}  X_unb_cb {X_unb_cb.shape}")

    # ----------------------------------------------------------------------
    # 1b. Load MolFormer-768 -- already row-aligned to load_test()
    # ----------------------------------------------------------------------
    X_te_mf = np.load(DATA_PROCESSED / "X_molformer_xl_te.npy").astype(np.float64)
    assert X_te_mf.shape[0] == n_test, f"molformer rows {X_te_mf.shape[0]} != n_test {n_test}"
    X_unb_mf = X_te_mf[unb_idx]
    print(f"[molformer] X_te_mf {X_te_mf.shape}  X_unb_mf {X_unb_mf.shape}")

    # ----------------------------------------------------------------------
    # 1c. Concatenate -> 1152-dim
    # ----------------------------------------------------------------------
    X_te = np.concatenate([X_te_cb, X_te_mf], axis=1)
    X_unb = np.concatenate([X_unb_cb, X_unb_mf], axis=1)
    assert X_te.shape == (n_test, 1152) and X_unb.shape == (n_unb, 1152)
    print(f"[concat] X_te {X_te.shape}  X_unb {X_unb.shape}  dim=1152")
    summary["dim_chemberta"] = 384
    summary["dim_molformer"] = 768
    summary["dim_total"] = 1152

    # ----------------------------------------------------------------------
    # 2. chemprop_aux anchor + residual
    # ----------------------------------------------------------------------
    te_anchor = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    anchor_unb = te_anchor[unb_idx]
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    residual = y_unb - anchor_unb
    print(f"[anchor] chemprop_aux unb_RAE = {rae_anchor_unb:.4f}  "
          f"residual std = {residual.std():.4f}")
    summary["chemprop_aux_unb_rae"] = rae_anchor_unb
    summary["residual_std"] = float(residual.std())

    # ----------------------------------------------------------------------
    # 3. Alpha sweep -- standalone OOF = chemprop_aux + r_hat
    # ----------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"RIDGE ALPHA SWEEP  alphas={ALPHA_GRID}  seeds={KF_SEEDS}")
    print("-" * 78)
    alpha_results = {}
    best = {"alpha": None, "mean_rae": float("inf"), "oof_mean": None, "r_hat_mean": None}
    for alpha in ALPHA_GRID:
        seed_raes, oofs, rhats = [], [], []
        for kf_seed in KF_SEEDS:
            r_hat = ridge_cv(X_unb, residual, unb_scaffolds, alpha, kf_seed)
            final_oof = anchor_unb + r_hat
            seed_raes.append(float(rae(y_unb, final_oof)))
            oofs.append(final_oof)
            rhats.append(r_hat)
        oof_mean = np.mean(np.column_stack(oofs), axis=1)
        rhat_mean = np.mean(np.column_stack(rhats), axis=1)
        mean_r = float(np.mean(seed_raes))
        std_r = float(np.std(seed_raes))
        rae_of_mean = float(rae(y_unb, oof_mean))
        alpha_results[str(alpha)] = {
            "alpha": float(alpha),
            "per_seed_rae": [float(x) for x in seed_raes],
            "mean_rae_seeds": mean_r,
            "std_rae_seeds": std_r,
            "rae_of_mean_oof": rae_of_mean,
        }
        marker = ""
        if mean_r < best["mean_rae"]:
            best = {
                "alpha": float(alpha),
                "mean_rae": mean_r,
                "oof_mean": oof_mean,
                "r_hat_mean": rhat_mean,
            }
            marker = "  <-- best so far"
        print(f"   alpha={alpha:>7.2f}  mean_RAE={mean_r:.4f} +/- {std_r:.4f}"
              f"  rae_of_mean={rae_of_mean:.4f}{marker}")

    summary["per_alpha_results"] = alpha_results
    summary["best_alpha"] = best["alpha"]
    summary["best_alpha_mean_rae"] = best["mean_rae"]
    summary["delta_vs_chemprop_aux"] = best["mean_rae"] - rae_anchor_unb
    summary["delta_vs_nb2240_K20"] = best["mean_rae"] - NB2240_REF

    np.save(DATA_PROCESSED / f"oof_{TAG}_resid_ridge.npy",
            best["oof_mean"].astype(np.float32))
    summary["standalone_oof_path"] = str(DATA_PROCESSED / f"oof_{TAG}_resid_ridge.npy")
    print(f"\n[best] alpha={best['alpha']}  mean_RAE={best['mean_rae']:.4f}")
    print(f"[delta] vs chemprop_aux ({rae_anchor_unb:.4f}) = "
          f"{best['mean_rae'] - rae_anchor_unb:+.4f}")
    print(f"[delta] vs nb2240_K20    ({NB2240_REF:.4f}) = "
          f"{best['mean_rae'] - NB2240_REF:+.4f}")

    # ----------------------------------------------------------------------
    # 4. Deploy refit (all 253 -> predict residual on 513)
    # ----------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("DEPLOY refit on all 253 -> predict residual on 513")
    print("-" * 78)
    sc_all = StandardScaler().fit(X_unb)
    Xunb_s = sc_all.transform(X_unb)
    Xte_s = sc_all.transform(X_te)
    m_deploy = Ridge(alpha=best["alpha"], random_state=0).fit(Xunb_s, residual)
    te_resid = m_deploy.predict(Xte_s)
    deploy_te = (te_anchor + te_resid).astype(np.float32)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_te)
    summary["te_path"] = str(DATA_PROCESSED / f"te_{TAG}.npy")
    in_rae_deploy = float(rae(y_unb, deploy_te[unb_idx]))
    summary["te_unb_in_sample_rae"] = in_rae_deploy
    summary["deploy_te_mean"] = float(deploy_te.mean())
    summary["deploy_te_std"] = float(deploy_te.std())
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {in_rae_deploy:.4f}")

    # ----------------------------------------------------------------------
    # 5. SLSQP blend gate
    # ----------------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"SLSQP BLEND GATE  (standalone {best['mean_rae']:.4f} "
          f"<= {STANDALONE_GATE}?)")
    print("-" * 78)
    if best["mean_rae"] <= STANDALONE_GATE:
        print("[gate] PASS  running 2-way + 3-way SLSQP")
        ensemble_col = best["oof_mean"]
        # nb2240 K=20 OOF on 253
        nb2240_oof = np.load(DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy").astype(np.float64)
        assert nb2240_oof.shape == (n_unb,), f"nb2240 OOF shape {nb2240_oof.shape}"
        nb2240_te = np.load(DATA_PROCESSED / "te_nb2240_K20.npy").astype(np.float64)

        # 2-way: {chemprop_aux, ensemble}
        P2 = np.column_stack([anchor_unb, ensemble_col])
        rae2_seeds, w2_seeds = [], []
        for kf_seed in KF_SEEDS:
            r2, _, fw = blend_cv(P2, y_unb, unb_scaffolds, kf_seed)
            rae2_seeds.append(r2)
            w2_seeds.append(np.mean(np.stack(fw), axis=0))
        rae2_mean = float(np.mean(rae2_seeds))
        w2_deploy = slsqp_simplex(P2, y_unb)
        te_2way = (w2_deploy[0] * te_anchor + w2_deploy[1] * deploy_te).astype(np.float32)

        # 3-way: {chemprop_aux, ensemble, nb2240}
        P3 = np.column_stack([anchor_unb, ensemble_col, nb2240_oof])
        rae3_seeds, w3_seeds = [], []
        for kf_seed in KF_SEEDS:
            r3, _, fw = blend_cv(P3, y_unb, unb_scaffolds, kf_seed)
            rae3_seeds.append(r3)
            w3_seeds.append(np.mean(np.stack(fw), axis=0))
        rae3_mean = float(np.mean(rae3_seeds))
        w3_deploy = slsqp_simplex(P3, y_unb)
        te_3way = (w3_deploy[0] * te_anchor
                   + w3_deploy[1] * deploy_te
                   + w3_deploy[2] * nb2240_te).astype(np.float32)

        print(f"   2-way per-seed: {[f'{r:.4f}' for r in rae2_seeds]}")
        print(f"   2-way mean RAE = {rae2_mean:.4f}  weights={w2_deploy.round(4).tolist()}")
        print(f"   3-way per-seed: {[f'{r:.4f}' for r in rae3_seeds]}")
        print(f"   3-way mean RAE = {rae3_mean:.4f}  weights={w3_deploy.round(4).tolist()}")

        best_blend_mean = min(rae2_mean, rae3_mean)
        if rae3_mean <= rae2_mean:
            chosen_te = te_3way
            chosen_w = w3_deploy.tolist()
            chosen_name = "3way_chemprop_ensemble_nb2240"
        else:
            chosen_te = te_2way
            chosen_w = w2_deploy.tolist()
            chosen_name = "2way_chemprop_ensemble"

        blend_passes = best_blend_mean < (NB2240_REF - BLEND_BEAT_MARGIN)
        np.save(DATA_PROCESSED / f"te_{TAG}_blend.npy", chosen_te)
        in_rae_blend = float(rae(y_unb, chosen_te[unb_idx]))

        summary["blend"] = {
            "gate": "PASS",
            "two_way": {
                "per_seed_rae": [float(r) for r in rae2_seeds],
                "mean_rae": rae2_mean,
                "deploy_weights_chemprop_ensemble": w2_deploy.tolist(),
            },
            "three_way": {
                "per_seed_rae": [float(r) for r in rae3_seeds],
                "mean_rae": rae3_mean,
                "deploy_weights_chemprop_ensemble_nb2240": w3_deploy.tolist(),
            },
            "chosen": chosen_name,
            "chosen_deploy_weights": chosen_w,
            "chosen_blend_te_path": str(DATA_PROCESSED / f"te_{TAG}_blend.npy"),
            "chosen_in_sample_unb_rae": in_rae_blend,
            "best_blend_mean_rae": best_blend_mean,
            "delta_vs_nb2240_K20": best_blend_mean - NB2240_REF,
            "deploy_csv_written": False,
        }

        if blend_passes:
            print(f"[gate] BLEND BEATS nb2240 by {NB2240_REF - best_blend_mean:.4f} "
                  f"(margin>= {BLEND_BEAT_MARGIN})  writing deploy CSV")
            sub_path = SUBMISSIONS / f"{TAG}_ensemble_embed.csv"
            sub_df = pd.DataFrame({
                "SMILES": te_smiles,
                "Molecule Name": te_names,
                "pEC50": chosen_te,
            })
            sub_df.to_csv(sub_path, index=False)
            summary["blend"]["deploy_csv_written"] = True
            summary["blend"]["deploy_csv_path"] = str(sub_path)
        else:
            print(f"[gate] BLEND mean {best_blend_mean:.4f} does NOT beat "
                  f"nb2240 ({NB2240_REF:.4f}) by {BLEND_BEAT_MARGIN}; no CSV")
    else:
        print(f"[gate] FAIL  standalone {best['mean_rae']:.4f} > {STANDALONE_GATE}")
        summary["blend"] = {
            "gate": "FAIL",
            "reason": f"standalone ridge {best['mean_rae']:.4f} > gate {STANDALONE_GATE}",
        }

    # ----------------------------------------------------------------------
    # 6. Final verdict
    # ----------------------------------------------------------------------
    if best["mean_rae"] < NB2240_REF - BLEND_BEAT_MARGIN:
        verdict = "STANDALONE_BEATS_NB2240"
    elif abs(best["mean_rae"] - NB2240_REF) <= BLEND_BEAT_MARGIN:
        verdict = "STANDALONE_FLAT_VS_NB2240"
    elif best["mean_rae"] < rae_anchor_unb - BLEND_BEAT_MARGIN:
        verdict = "STANDALONE_BEATS_CHEMPROP_BUT_BELOW_NB2240"
    else:
        verdict = "STANDALONE_NO_GAIN"
    summary["verdict_standalone"] = verdict
    print(f"\n[verdict] standalone: {verdict}")

    summary["wall_sec"] = round(time.time() - t0, 2)
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print("=" * 78)
    print(f"  standalone best alpha: {best['alpha']}")
    print(f"  standalone mean RAE:   {best['mean_rae']:.4f}")
    print(f"  chemprop_aux RAE:      {rae_anchor_unb:.4f}")
    print(f"  nb2240 K=20 ref RAE:   {NB2240_REF:.4f}")
    print(f"  delta vs nb2240:       {best['mean_rae'] - NB2240_REF:+.4f}")
    if summary["blend"].get("gate") == "PASS":
        print(f"  blend best:            {summary['blend']['best_blend_mean_rae']:.4f}")
        print(f"  chosen blend:          {summary['blend']['chosen']}")
    print(f"  verdict:               {verdict}")
    print(f"  wall:                  {summary['wall_sec']}s")

    return summary


if __name__ == "__main__":
    main()

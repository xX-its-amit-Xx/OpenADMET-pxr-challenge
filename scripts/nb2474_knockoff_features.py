"""
nb2474 - Knockoff filter (Barber-Candes) for valid feature selection on K=20 RFE.

Knockoff filter generates "knockoff" copies of each feature that match marginal
distribution but are conditionally independent of the response, then uses a
lasso feature statistic + Barber-Candes selective FDR procedure to select a
provably-FDR-controlled subset of features. Selected features feed a small
LGBM that predicts the residual of the nb2240 anchor.

SUBSTRATE:
  - X_117_unb.npy (253, 117)  - pyramid candidate features (cycle 196)
  - X_117_te.npy  (513, 117)  - test pyramid features
  - nb2240_mean_bag_oof_K20.npy (253,) - anchor OOF
  - te_nb2240_K20.npy (513,) - anchor deploy
  - _audit_unblind_y.npy (253,) - unblind truth

PROTOCOL:
  1) KnockoffFilter(ksampler='gaussian', fstat='lasso', fdr=0.20)
     on (X_117_unb, residual = y_unb - nb2240_oof)
  2) Take selected feature indices.
  3) LGBM(max_depth=4, n_est=300, lr=0.03, min_child=5) on selected features
     under 5-fold scaffold CV (residual target).
  4) Final prediction = nb2240 anchor + lgbm residual prediction.
  5) Gate: mean_rae < 0.4570 -> PROMOTE, else FAIL.

OUTPUTS:
  - data/processed/nb2474_summary.json
  - data/processed/nb2474_pred_oof.npy        (length 253)
  - data/processed/te_nb2474.npy              (length 513)
  - data/processed/nb2474_selected_features.json
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"
SUMMARY_PATH = PROC / "nb2474_summary.json"


def _write_summary(payload: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _abort_install_failed(detail: str) -> None:
    payload = {
        "script": "nb2474_knockoff_features",
        "status": "INSTALL_FAILED",
        "reason": "knockpy import failed; package not installable in this env",
        "detail": detail[:2000],
        "gate": "FAIL",
        "promote": False,
    }
    _write_summary(payload)
    print(json.dumps(payload, indent=2))
    sys.exit(0)


def main() -> None:
    # ---- Hard import of knockpy; if it fails, save INSTALL_FAILED and exit. ----
    try:
        import knockpy  # noqa: F401
        from knockpy import KnockoffFilter
    except Exception as exc:  # noqa: BLE001
        _abort_install_failed(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        )
        return  # pragma: no cover

    import lightgbm as lgb
    import pandas as pd

    sys.path.insert(0, str(ROOT / "src"))
    from pxr.data import load_train  # noqa: E402
    from pxr.eval import rae, scaffold_kfold_indices  # noqa: E402

    rng = np.random.default_rng(20260608)

    # ---- Load substrate ----
    X_unb = np.load(PROC / "pyramid" / "X_117_unb.npy")  # (253, 117)
    X_te = np.load(PROC / "pyramid" / "X_117_te.npy")    # (513, 117)
    anchor_oof = np.load(PROC / "nb2240_mean_bag_oof_K20.npy")  # (253,)
    anchor_te = np.load(PROC / "te_nb2240_K20.npy")             # (513,)
    y_unb = np.load(PROC / "_audit_unblind_y.npy").astype(float)  # (253,)

    assert X_unb.shape == (253, 117), f"X_unb shape {X_unb.shape}"
    assert X_te.shape == (513, 117), f"X_te shape {X_te.shape}"
    assert anchor_oof.shape == (253,), f"anchor_oof shape {anchor_oof.shape}"
    assert anchor_te.shape == (513,), f"anchor_te shape {anchor_te.shape}"
    assert y_unb.shape == (253,), f"y_unb shape {y_unb.shape}"

    residual = y_unb - anchor_oof

    # ---- Standardize X for gaussian knockoffs ----
    mu = X_unb.mean(axis=0, keepdims=True)
    sd = X_unb.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Xs_unb = (X_unb - mu) / sd
    Xs_te = (X_te - mu) / sd

    # Drop zero-variance columns (knockoff Sigma must be PD-ish).
    keep_cols = np.where(sd.ravel() > 1e-8)[0]
    Xs_unb_k = Xs_unb[:, keep_cols]
    p_kept = Xs_unb_k.shape[1]

    # ---- Knockoff filter ----
    kf = KnockoffFilter(ksampler="gaussian", fstat="lasso")
    try:
        kf.forward(X=Xs_unb_k, y=residual, fdr=0.20)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "script": "nb2474_knockoff_features",
            "status": "FAIL",
            "reason": f"KnockoffFilter.forward failed: {type(exc).__name__}: {exc}",
            "gate": "FAIL",
            "promote": False,
        }
        _write_summary(payload)
        print(json.dumps(payload, indent=2))
        return

    selections = np.asarray(kf.selected_flags).astype(bool)
    selected_local = np.where(selections)[0]
    selected_global = keep_cols[selected_local].tolist()
    n_selected = len(selected_global)

    # If nothing selected, abort gracefully.
    if n_selected == 0:
        payload = {
            "script": "nb2474_knockoff_features",
            "status": "FAIL",
            "reason": "knockoff selected 0 features at fdr=0.20",
            "n_candidates": int(p_kept),
            "gate": "FAIL",
            "promote": False,
        }
        _write_summary(payload)
        print(json.dumps(payload, indent=2))
        return

    # ---- Scaffold-CV LGBM on the selected residual subset ----
    train_df = load_train()
    # The 253 unblind rows live as a subset of train; map via Molecule Name -> scaffold.
    # We re-derive scaffold split using the unblind compound names if available.
    unb_names_path = PROC / "_audit_unblind_names.npy"
    if unb_names_path.exists():
        unb_names = np.load(unb_names_path, allow_pickle=True)
        name_col = (
            "molecule_name"
            if "molecule_name" in train_df.columns
            else ("name" if "name" in train_df.columns else None)
        )
        if name_col is None:
            scaf_for_unb = np.arange(253)  # fallback: random folds
        else:
            scaf_lookup = dict(zip(train_df[name_col].astype(str), train_df.get("scaffold", pd.Series(["?"] * len(train_df))).astype(str)))
            scaf_for_unb = np.array([scaf_lookup.get(str(n), f"_{i}") for i, n in enumerate(unb_names)])
    else:
        scaf_for_unb = np.array([f"s{i}" for i in range(253)])

    folds = scaffold_kfold_indices(scaf_for_unb, n_splits=5, seed=42)

    Xsel_unb = X_unb[:, selected_global].astype(np.float32)
    Xsel_te = X_te[:, selected_global].astype(np.float32)

    resid_oof = np.zeros(253, dtype=np.float64)
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.1,
            random_state=42 + fold_i,
            verbose=-1,
            n_jobs=2,
        )
        model.fit(Xsel_unb[tr_idx], residual[tr_idx])
        resid_oof[va_idx] = model.predict(Xsel_unb[va_idx])

    pred_oof = anchor_oof + resid_oof

    # ---- Deploy refit on all 253 for te output ----
    final_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.03,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=2,
    )
    final_model.fit(Xsel_unb, residual)
    te_resid = final_model.predict(Xsel_te)
    te_pred = anchor_te + te_resid

    # ---- Scoring ----
    score_rae = float(rae(y_unb, pred_oof))
    anchor_rae = float(rae(y_unb, anchor_oof))
    delta = score_rae - anchor_rae

    gate_threshold = 0.4570
    promote = score_rae < gate_threshold
    gate = "PROMOTE" if promote else "FAIL"

    # ---- Save artifacts ----
    np.save(PROC / "nb2474_pred_oof.npy", pred_oof.astype(np.float32))
    np.save(PROC / "te_nb2474.npy", te_pred.astype(np.float32))
    with open(PROC / "nb2474_selected_features.json", "w") as fh:
        json.dump(
            {
                "n_selected": n_selected,
                "selected_indices_global": selected_global,
                "fdr_target": 0.20,
                "p_total": int(X_unb.shape[1]),
                "p_kept_for_knockoff": int(p_kept),
            },
            fh,
            indent=2,
        )

    payload = {
        "script": "nb2474_knockoff_features",
        "status": "OK",
        "anchor": "nb2240_K20",
        "fdr_target": 0.20,
        "n_features_total": int(X_unb.shape[1]),
        "n_features_kept_for_knockoff": int(p_kept),
        "n_features_selected": int(n_selected),
        "selected_indices_preview": selected_global[:20],
        "mean_rae_oof": score_rae,
        "anchor_rae_oof": anchor_rae,
        "delta_vs_anchor": delta,
        "gate_threshold": gate_threshold,
        "gate": gate,
        "promote": bool(promote),
        "outputs": {
            "pred_oof": "data/processed/nb2474_pred_oof.npy",
            "te": "data/processed/te_nb2474.npy",
            "selected_features": "data/processed/nb2474_selected_features.json",
        },
    }
    _write_summary(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

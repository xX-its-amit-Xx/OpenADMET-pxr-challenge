"""nb1050 -- PyTorch NN head with PROPER regularization on K=28 SHAP residual.

HYPOTHESIS:
    Cycle 126 nb2191 sklearn MLP(64,32) hit RAE 0.4737/0.4698 baseline floor --
    it failed to extract orthogonal signal from chemprop_aux because:
       * alpha=0.01 is weak L2 (no batch-norm, no dropout, no early-stopping)
       * sklearn MLPRegressor lacks modern regularization tooling
       * single-loss MSE only, no LR schedule
    This notebook tests a PyTorch MLP with the proper bag of tricks:
       * BatchNorm1d after each hidden layer (covariate-shift reg)
       * Dropout(0.3) then Dropout(0.2) (stochastic reg)
       * AdamW with heavy weight_decay=1e-2 (decoupled L2)
       * Cosine LR schedule over 100 epochs
       * Early stopping on validation MSE with patience=10
    Cross-paradigm test against nb2103 K=28 LGBM at decision_margin 0.003.

PROTOCOL:
    1. Load:
         - chemprop_aux predictions on 253 (anchor)
         - cached SHAP top-28 feature matrix X_unb_28_nb2103 (253, 28)
         - nb2103 K=28 LGBM mean-bag OOF (blend axis)
         - truth y_unb (253,)
    2. Build PyTorch MLP:
         28 -> Linear(64) -> BatchNorm1d -> ReLU -> Dropout(0.3)
            -> Linear(32) -> BatchNorm1d -> ReLU -> Dropout(0.2)
            -> Linear(1)
    3. Loss: MSE; Optim: AdamW lr=1e-3 weight_decay=1e-2
    4. LR schedule: CosineAnnealingLR over 100 epochs
    5. Early stop: patience=10 epochs on validation MSE; restore best weights
    6. Standardize features with StandardScaler fit INSIDE each fold
    7. 5-seed bag (seeds 0, 1, 7, 42, 137), 5-fold cross-fit on residual
       (y_unb - chemprop_aux). Inner train/val split (80/20 of fold-train)
       used for early stopping.
    8. Final = chemprop_aux + cross-fit-NN-residual.
       Report mean-bag and median-bag RAE.
    9. NN-vs-LGBM convex-blend sweep w in {0.0, 0.25, 0.5, 0.75, 1.0}
    10. Compare vs nb2103 K=28 (mean_bag 0.4737, median_bag 0.4698) at
        decision_margin 0.003.
    11. If best beats: build deploy CSV (chemprop_aux base + NN residual on
        253 unb rows only; blind 260 left at chemprop_aux raw).

Outputs:
    scripts/nb1050_nn_head.py
    data/processed/nb1050_summary.json
    data/processed/nb1050_nn_mean_bag_oof.npy   (253,) float32
    data/processed/nb1050_nn_median_bag_oof.npy (253,) float32
    submissions/nb1050_nn_head.csv  (only if beats by >=0.003)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS as SUBMISSIONS_DIR

TAG = "nb1050"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X_K28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]

# NN hyperparams (regularization-heavy)
NN_HIDDEN_1 = 64
NN_HIDDEN_2 = 32
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-2
N_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 32
INNER_VAL_FRAC = 0.20

# References
NB2103_K28_MEAN_BAG_REF = 0.4737
NB2103_K28_MEDIAN_BAG_REF = 0.4698
CHEMPROP_AUX_REF = 0.6216
DECISION_MARGIN = 0.003


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RegMLP(nn.Module):
    """28 -> 64 BN ReLU Dropout(0.3) -> 32 BN ReLU Dropout(0.2) -> 1."""

    def __init__(self, in_dim: int = 28):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, NN_HIDDEN_1)
        self.bn1 = nn.BatchNorm1d(NN_HIDDEN_1)
        self.do1 = nn.Dropout(DROPOUT_1)
        self.fc2 = nn.Linear(NN_HIDDEN_1, NN_HIDDEN_2)
        self.bn2 = nn.BatchNorm1d(NN_HIDDEN_2)
        self.do2 = nn.Dropout(DROPOUT_2)
        self.head = nn.Linear(NN_HIDDEN_2, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.do1(self.act(self.bn1(self.fc1(x))))
        h = self.do2(self.act(self.bn2(self.fc2(h))))
        return self.head(h).squeeze(-1)


def _train_one_model(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va_inner: np.ndarray, y_va_inner: np.ndarray,
    seed: int,
):
    """Train one MLP with early stopping. Return best-weights model + log."""
    _set_seed(seed)
    device = torch.device("cpu")  # CPU per env
    model = RegMLP(in_dim=X_tr.shape[1]).to(device)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_tr.astype(np.float32))
    yt = torch.from_numpy(y_tr.astype(np.float32))
    Xv = torch.from_numpy(X_va_inner.astype(np.float32))
    yv = torch.from_numpy(y_va_inner.astype(np.float32))

    # DataLoader with bsz=32 (drop_last=False; if BN gets bsz=1 it errors,
    # so we floor batch size based on data size).
    bs = min(BATCH_SIZE, max(2, len(X_tr) // 2))
    ds = TensorDataset(Xt, yt)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, generator=g,
                        drop_last=(len(X_tr) % bs == 1))

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = PATIENCE
    train_losses, val_losses = [], []
    for epoch in range(N_EPOCHS):
        model.train()
        ep_loss = 0.0
        n_batch = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batch += 1
        sched.step()
        ep_loss = ep_loss / max(1, n_batch)
        train_losses.append(ep_loss)

        model.eval()
        with torch.no_grad():
            v_pred = model(Xv)
            v_loss = float(loss_fn(v_pred, yv).item())
        val_losses.append(v_loss)

        if v_loss < best_val - 1e-6:
            best_val = v_loss
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_val_mse": best_val,
        "best_epoch": int(best_epoch),
        "final_epoch": int(epoch),
        "early_stopped": bool(patience_left <= 0),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_val_loss": float(val_losses[-1]) if val_losses else None,
    }


def _predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X.astype(np.float32))).numpy()
    return pred.astype(np.float64)


def _nn_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One seed: 5-fold cross-fit NN on standardized features w/ inner ES."""
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_logs: list[dict] = []
    for fi, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        # Inner train/val split for early stopping
        inner_tr_loc, inner_val_loc = train_test_split(
            tr_loc, test_size=INNER_VAL_FRAC, random_state=seed,
        )
        sc = StandardScaler()
        X_inner_tr = sc.fit_transform(X[inner_tr_loc])
        X_inner_val = sc.transform(X[inner_val_loc])
        X_va = sc.transform(X[va_loc])
        y_inner_tr = residual[inner_tr_loc]
        y_inner_val = residual[inner_val_loc]
        model, log = _train_one_model(
            X_inner_tr, y_inner_tr, X_inner_val, y_inner_val, seed,
        )
        log["fold"] = int(fi)
        log["n_inner_tr"] = int(len(inner_tr_loc))
        log["n_inner_val"] = int(len(inner_val_loc))
        log["n_va_outer"] = int(len(va_loc))
        fold_logs.append(log)
        oof[va_loc] = _predict(model, X_va)
    return oof, fold_logs


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PyTorch NN head (REG-HEAVY) residual on K=28 SHAP")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          arch: 28->64 BN ReLU Dropout({DROPOUT_1}) "
          f"->32 BN ReLU Dropout({DROPOUT_2}) ->1")
    print(f"          optim: AdamW lr={LR} wd={WEIGHT_DECAY}  "
          f"sched: CosineAnnealingLR(T_max={N_EPOCHS})")
    print(f"          early-stop: patience={PATIENCE}  bs={BATCH_SIZE}  "
          f"inner_val={INNER_VAL_FRAC}")
    print(f"          ref: nb2103 K=28 mean_bag={NB2103_K28_MEAN_BAG_REF:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor ----
    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load K=28 SHAP feature matrix ----
    if not X_K28_PATH.exists():
        raise FileNotFoundError(
            f"missing K=28 feature cache {X_K28_PATH} -- run nb2103 first"
        )
    X_unb = np.load(X_K28_PATH).astype(np.float32)
    if X_unb.shape != (n_unb, 28):
        raise ValueError(
            f"K=28 feature matrix shape mismatch: {X_unb.shape} != "
            f"({n_unb}, 28)"
        )
    print(f"[feat] X_unb (K=28) shape = {X_unb.shape}  "
          f"mean = {X_unb.mean():+.4f}  std = {X_unb.std():.4f}")

    # ---- Load nb2103 K=28 LGBM OOF (for blend axis) ----
    if not NB2103_K28_OOF_PATH.exists():
        raise FileNotFoundError(
            f"missing {NB2103_K28_OOF_PATH} -- run nb2103 first"
        )
    lgbm_mean_bag_oof = np.load(NB2103_K28_OOF_PATH).astype(np.float64)
    rae_lgbm_mean_bag = float(rae(y_unb, lgbm_mean_bag_oof))
    print(f"[load] nb2103 K=28 LGBM mean-bag OOF rae = {rae_lgbm_mean_bag:.4f}  "
          f"(ref {NB2103_K28_MEAN_BAG_REF:.4f})")

    # ---- 5-seed bag NN residual cross-fit ----
    print("\n" + "-" * 78)
    print("5-SEED NN RESIDUAL CROSS-FIT")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_resid_oof = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records: list[dict] = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s, fold_logs = _nn_cross_fit_one_seed(X_unb, residual, s)
        per_seed_resid_oof[i] = resid_oof_s
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        es_count = sum(1 for fl in fold_logs if fl["early_stopped"])
        mean_best_epoch = float(
            np.mean([fl["best_epoch"] for fl in fold_logs])
        )
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_anchor": delta_s,
            "resid_oof_mean": float(resid_oof_s.mean()),
            "resid_oof_std": float(resid_oof_s.std()),
            "n_early_stopped_folds": int(es_count),
            "mean_best_epoch": mean_best_epoch,
            "fold_logs": fold_logs,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed={s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"resid_std = {resid_oof_s.std():.4f}  "
              f"ES_folds={es_count}/5  "
              f"mean_best_ep={mean_best_epoch:.1f}  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    print(f"\n   per-seed RAE  = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean = {per_seed_rae_arr.mean():.4f}  "
          f"std = {per_seed_rae_arr.std():.4f}")
    print(f"   NN mean-bag   RAE = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}  "
          f"d_vs_K28meanbag = "
          f"{rae_mean_bag - NB2103_K28_MEAN_BAG_REF:+.4f}  "
          f"d_vs_K28medianbag = "
          f"{rae_mean_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")
    print(f"   NN median-bag RAE = {rae_median_bag:.4f}  "
          f"(d_vs_anchor = {rae_median_bag - rae_anchor:+.4f}  "
          f"d_vs_K28meanbag = "
          f"{rae_median_bag - NB2103_K28_MEAN_BAG_REF:+.4f}  "
          f"d_vs_K28medianbag = "
          f"{rae_median_bag - NB2103_K28_MEDIAN_BAG_REF:+.4f})")

    # ---- Save OOFs ----
    out_nn_mean = DATA_PROCESSED / f"{TAG}_nn_mean_bag_oof.npy"
    out_nn_med = DATA_PROCESSED / f"{TAG}_nn_median_bag_oof.npy"
    np.save(out_nn_mean, mean_bag_oof.astype(np.float32))
    np.save(out_nn_med, median_bag_oof.astype(np.float32))
    print(f"   [save] {out_nn_mean}")
    print(f"   [save] {out_nn_med}")

    # ---- NN+LGBM convex-weight blend sweep ----
    print("\n" + "-" * 78)
    print("NN + LGBM (nb2103 K=28) CONVEX BLEND SWEEP")
    print("-" * 78)
    blend_records: list[dict] = []
    for w in BLEND_WEIGHTS:
        blend = w * mean_bag_oof + (1.0 - w) * lgbm_mean_bag_oof
        rae_w = float(rae(y_unb, blend))
        d_vs_lgbm_mean = rae_w - rae_lgbm_mean_bag
        d_vs_lgbm_med = rae_w - NB2103_K28_MEDIAN_BAG_REF
        print(f"   w_NN={w:.2f}  w_LGBM={1.0 - w:.2f}  "
              f"blend RAE = {rae_w:.4f}  "
              f"d_vs_LGBMmean = {d_vs_lgbm_mean:+.4f}  "
              f"d_vs_LGBMmed = {d_vs_lgbm_med:+.4f}")
        blend_records.append({
            "w_nn": float(w),
            "w_lgbm": float(1.0 - w),
            "rae_blend": rae_w,
            "delta_vs_lgbm_mean_bag": d_vs_lgbm_mean,
            "delta_vs_lgbm_median_bag": d_vs_lgbm_med,
        })
    best_blend_i = int(np.argmin([r["rae_blend"] for r in blend_records]))
    best_blend = blend_records[best_blend_i]
    print(f"\n   best blend = w_NN={best_blend['w_nn']:.2f}  "
          f"RAE = {best_blend['rae_blend']:.4f}")

    # ---- Verdicts ----
    print("\n" + "=" * 78)
    print("VERDICTS")
    print("=" * 78)
    candidates: dict[str, float] = {
        "nn_mean_bag": rae_mean_bag,
        "nn_median_bag": rae_median_bag,
        "best_blend": best_blend["rae_blend"],
    }
    print(f"   nb2103 K=28 mean_bag   ref = {NB2103_K28_MEAN_BAG_REF:.4f}")
    print(f"   nb2103 K=28 median_bag ref = {NB2103_K28_MEDIAN_BAG_REF:.4f}  "
          f"<-- target to beat")
    for name, val in candidates.items():
        d_vs_med = val - NB2103_K28_MEDIAN_BAG_REF
        d_vs_mean = val - NB2103_K28_MEAN_BAG_REF
        if val < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN:
            v = "BEATS_NB2103_K28_MEDIAN_BAG"
        elif val < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN:
            v = "BEATS_NB2103_K28_MEAN_BAG_ONLY"
        elif abs(d_vs_mean) < DECISION_MARGIN:
            v = "FLAT_VS_NB2103_K28_MEAN_BAG"
        else:
            v = "HURTS_VS_NB2103_K28"
        print(f"   {name:>18s}  RAE = {val:.4f}  "
              f"d_vs_meanbag = {d_vs_mean:+.4f}  "
              f"d_vs_medianbag = {d_vs_med:+.4f}  -> {v}")

    best_candidate_name = min(candidates, key=lambda k: candidates[k])
    best_candidate_rae = candidates[best_candidate_name]
    beats_median_bag = (
        best_candidate_rae < NB2103_K28_MEDIAN_BAG_REF - DECISION_MARGIN
    )
    beats_mean_bag = (
        best_candidate_rae < NB2103_K28_MEAN_BAG_REF - DECISION_MARGIN
    )

    if beats_median_bag:
        global_verdict = (
            f"NN_HEAD_BEATS_NB2103_K28_MEDIAN_BAG  "
            f"({best_candidate_name} = {best_candidate_rae:.4f})"
        )
    elif beats_mean_bag:
        global_verdict = (
            f"NN_HEAD_BEATS_ONLY_NB2103_K28_MEAN_BAG  "
            f"({best_candidate_name} = {best_candidate_rae:.4f})"
        )
    elif abs(best_candidate_rae - NB2103_K28_MEAN_BAG_REF) < DECISION_MARGIN:
        global_verdict = (
            f"NN_HEAD_FLAT_VS_NB2103_K28_MEAN_BAG  "
            f"({best_candidate_name} = {best_candidate_rae:.4f})"
        )
    else:
        global_verdict = (
            f"NN_HEAD_HURTS_VS_NB2103_K28  "
            f"({best_candidate_name} = {best_candidate_rae:.4f})"
        )
    print(f"\n   global verdict = {global_verdict}")

    # ---- Deploy CSV (only if beats by margin) ----
    deploy_csv_path = None
    if beats_median_bag:
        print("\n   [deploy] NN head beats target by >=0.003 -- "
              "building deploy CSV")
        te_smiles = te["smiles"].astype(str).tolist() \
            if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
        if "Molecule Name" in te.columns:
            te_names = te["Molecule Name"].astype(str).tolist()
        elif "molecule_name" in te.columns:
            te_names = te["molecule_name"].astype(str).tolist()
        else:
            te_names = [f"Compound_{i}" for i in range(n_test)]
        deploy_pred_513 = te_anchor_513.copy()
        if best_candidate_name == "nn_mean_bag":
            deploy_pred_513[unb_idx] = mean_bag_oof
        elif best_candidate_name == "nn_median_bag":
            deploy_pred_513[unb_idx] = median_bag_oof
        else:  # best_blend
            w = best_blend["w_nn"]
            blend_unb = w * mean_bag_oof + (1.0 - w) * lgbm_mean_bag_oof
            deploy_pred_513[unb_idx] = blend_unb
        deploy_df = pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_pred_513,
        })
        deploy_csv_path = SUBMISSIONS_DIR / f"{TAG}_nn_head.csv"
        deploy_df.to_csv(deploy_csv_path, index=False)
        print(f"   [save] {deploy_csv_path}")
    else:
        print("\n   [deploy] best NN does NOT beat target by 0.003 -- "
              "no deploy CSV built")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": ("torch_mlp_reg_head_residual_on_K28_SHAP_features"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "feature_source": str(X_K28_PATH),
        "feature_K": 28,
        "model_family": "PyTorch_MLP_RegHeavy",
        "arch": "28->64 BN ReLU Dropout(0.3) ->32 BN ReLU Dropout(0.2) ->1",
        "nn_hidden_1": NN_HIDDEN_1,
        "nn_hidden_2": NN_HIDDEN_2,
        "dropout_1": DROPOUT_1,
        "dropout_2": DROPOUT_2,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": f"CosineAnnealingLR(T_max={N_EPOCHS})",
        "n_epochs_max": N_EPOCHS,
        "early_stop_patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "inner_val_frac": INNER_VAL_FRAC,
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "standardize_within_fold": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_records": per_seed_records,
        "per_seed_rae": per_seed_rae,
        "rae_per_seed_mean": float(per_seed_rae_arr.mean()),
        "rae_per_seed_median": float(np.median(per_seed_rae_arr)),
        "rae_per_seed_std": float(per_seed_rae_arr.std()),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb2103_K28_meanbag": (
            rae_mean_bag - NB2103_K28_MEAN_BAG_REF
        ),
        "delta_mean_bag_vs_nb2103_K28_medianbag": (
            rae_mean_bag - NB2103_K28_MEDIAN_BAG_REF
        ),
        "delta_median_bag_vs_nb2103_K28_meanbag": (
            rae_median_bag - NB2103_K28_MEAN_BAG_REF
        ),
        "delta_median_bag_vs_nb2103_K28_medianbag": (
            rae_median_bag - NB2103_K28_MEDIAN_BAG_REF
        ),
        "lgbm_mean_bag_rae_reproduced": rae_lgbm_mean_bag,
        "blend_weights": BLEND_WEIGHTS,
        "blend_records": blend_records,
        "best_blend_w_nn": best_blend["w_nn"],
        "best_blend_rae": best_blend["rae_blend"],
        "best_candidate_name": best_candidate_name,
        "best_candidate_rae": best_candidate_rae,
        "beats_nb2103_K28_mean_bag": bool(beats_mean_bag),
        "beats_nb2103_K28_median_bag": bool(beats_median_bag),
        "verdict": global_verdict,
        "deploy_csv": str(deploy_csv_path) if deploy_csv_path else None,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG_REF,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "pre_unblind_clean": True,
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
        "feature_K", "arch",
        "rae_anchor_chemprop_aux",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb2103_K28_medianbag",
        "delta_median_bag_vs_nb2103_K28_medianbag",
        "best_blend_w_nn", "best_blend_rae",
        "best_candidate_name", "best_candidate_rae",
        "beats_nb2103_K28_mean_bag", "beats_nb2103_K28_median_bag",
        "verdict", "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== BLEND TABLE ====")
    for r in res["blend_records"]:
        print(f"  w_NN={r['w_nn']:.2f}  RAE={r['rae_blend']:.4f}  "
              f"d_vs_LGBM_medianbag={r['delta_vs_lgbm_median_bag']:+.4f}")

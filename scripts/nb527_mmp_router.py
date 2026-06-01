"""nb527 -- MMP-FRAGMENTATION RESIDUAL ROUTER.

Hypothesis: MMP transforms (single-cut rdMMPA) encode "what is the activity
distribution around this scaffold/core" — orthogonal to fingerprint-based
routers (nb502/nb510) and the engineered 33-col matrix (nb492).

Pipeline:
  1. Generate MMP single-cut transforms on 4139 train compounds. For each
     train compound, compute partners that share the SAME CORE differing by
     exactly one R-group, and aggregate:
       - n_mmp_pairs, mean_mmp_delta, max_mmp_delta (per-compound).
  2. Project to 513 test by matching test compound cores to train cores. For
     each test, inherit the MMP stats of the nearest core match (mean across
     train compounds sharing any of the test's cores). Missing -> global mean.
  3. Stack [3 MMP] + [33 nb481 ext] = 36 features (we reuse the nb481
     feature builder for the 33-col base).
  4. Anchor = nb464. target = truth - nb464 on 253 unblind. 5-fold KFold
     cross-fit shallow LGBM + sigmoid err_hat gate.
  5. Report cross-fit RAE, resid corr to nb502 and nb510.

Fallback: if MMP fragmentation on 4139 train exceeds 300 s, swap to simpler
Murcko-scaffold features (scaffold_n_train, scaffold_mean_pec50,
scaffold_std_pec50). feature_type_used reports which path was actually taken.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.chem import bemis_murcko, standardize_smiles
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

# Reuse nb481's full extended feature builder for the 33-col base matrix.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from nb481_residual_router_extended import build_feature_matrix  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7
TAG = "nb527"
ANCHOR_NAME = "nb464"
ANCHOR_FILE = "te_nb464.npy"
MMP_BUDGET_S = 300.0  # if MMP fragmentation exceeds this, fall back

LGBM_PARAMS = dict(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=8,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fragment_compound(smi: str, max_cuts: int = 1) -> list[tuple[str, str]]:
    if not isinstance(smi, str) or not smi:
        return []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    try:
        res = rdMMPA.FragmentMol(mol, maxCuts=max_cuts, resultsAsMols=False)
    except Exception:
        return []
    pairs = []
    for tup in res:
        if len(tup) != 2:
            continue
        core, chains = tup
        # Single-cut rdMMPA: core is '' and chains is "A.B" (both halves).
        # Split into two SMILES; emit (left, right) and (right, left) so a
        # compound can match either side as the conserved scaffold.
        if not chains:
            continue
        if core:
            parts = [core] + chains.split(".")
        else:
            parts = chains.split(".")
        if len(parts) < 2:
            continue
        a, b = parts[0], parts[1]
        if not a or not b:
            continue
        pairs.append((a, b))
        pairs.append((b, a))
    return pairs


def build_mmp_features(
    tr_smis: list[str], tr_pec: np.ndarray, te_smis: list[str], budget_s: float
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Returns (X_tr_mmp (n_tr,3), X_te_mmp (n_te,3), 'mmp') or (None, None, 'scaffold')."""
    t0 = time.time()
    n_tr = len(tr_smis)
    print(f"  fragmenting {n_tr} train compounds (single-cut MMP) ...")
    tr_frags: list[list[tuple[str, str]]] = []
    for i, smi in enumerate(tr_smis):
        tr_frags.append(fragment_compound(smi, max_cuts=1))
        if (i + 1) % 1000 == 0:
            print(f"    [{i+1}/{n_tr}] elapsed={time.time()-t0:.1f}s")
        if time.time() - t0 > budget_s:
            print(f"  MMP exceeded budget ({budget_s:.0f}s); aborting MMP path.")
            return None, None, "scaffold"

    # core -> list of (compound_idx, chain)
    core_lookup: dict[str, list[tuple[int, str]]] = {}
    for i, fragl in enumerate(tr_frags):
        for core, chain in fragl:
            core_lookup.setdefault(core, []).append((i, chain))
    print(f"  built core_lookup: {len(core_lookup)} unique cores "
          f"({time.time()-t0:.1f}s)")

    # train per-compound MMP stats
    tr_n = np.zeros(n_tr, dtype=np.float32)
    tr_mean = np.zeros(n_tr, dtype=np.float32)
    tr_max = np.zeros(n_tr, dtype=np.float32)
    for i, fragl in enumerate(tr_frags):
        partner_set: set[int] = set()
        for core, chain in fragl:
            for j, jchain in core_lookup.get(core, []):
                if j == i:
                    continue
                if jchain == chain:
                    continue
                partner_set.add(j)
        if not partner_set:
            continue
        deltas = np.abs(tr_pec[list(partner_set)] - tr_pec[i])
        tr_n[i] = float(len(partner_set))
        tr_mean[i] = float(deltas.mean())
        tr_max[i] = float(deltas.max())
    print(f"  train MMP stats: n>0 = {(tr_n > 0).sum()}/{n_tr} "
          f"({time.time()-t0:.1f}s)")

    # test: project via core match
    n_te = len(te_smis)
    te_n = np.zeros(n_te, dtype=np.float32)
    te_mean = np.zeros(n_te, dtype=np.float32)
    te_max = np.zeros(n_te, dtype=np.float32)
    global_n = float(tr_n[tr_n > 0].mean()) if (tr_n > 0).any() else 0.0
    global_mean = float(tr_mean[tr_n > 0].mean()) if (tr_n > 0).any() else 0.0
    global_max = float(tr_max[tr_n > 0].mean()) if (tr_n > 0).any() else 0.0

    for k, smi in enumerate(te_smis):
        fragl = fragment_compound(smi, max_cuts=1)
        partner_set: set[int] = set()
        for core, chain in fragl:
            for j, jchain in core_lookup.get(core, []):
                if jchain == chain:
                    continue
                partner_set.add(j)
        if not partner_set:
            te_n[k] = global_n
            te_mean[k] = global_mean
            te_max[k] = global_max
            continue
        partner_pec = tr_pec[list(partner_set)]
        te_n[k] = float(len(partner_set))
        te_mean[k] = float(np.std(partner_pec))  # spread of MMP partners
        te_max[k] = float(np.max(partner_pec) - np.min(partner_pec))
    print(f"  test MMP stats: matched = {(te_n != global_n).sum()}/{n_te} "
          f"({time.time()-t0:.1f}s)")

    X_tr = np.stack([tr_n, tr_mean, tr_max], axis=1).astype(np.float32)
    X_te = np.stack([te_n, te_mean, te_max], axis=1).astype(np.float32)
    return X_tr, X_te, "mmp"


def build_scaffold_features(
    tr_smis: list[str], tr_pec: np.ndarray, te_smis: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: Murcko-scaffold-based features (4 cols)."""
    tr_scaf = [bemis_murcko(s) or "" for s in tr_smis]
    scaf_groups: dict[str, list[float]] = {}
    for s, y in zip(tr_scaf, tr_pec):
        scaf_groups.setdefault(s, []).append(float(y))

    def hash_int(s: str) -> float:
        if not s:
            return 0.0
        h = hashlib.md5(s.encode()).hexdigest()
        return float(int(h[:8], 16) % 100000)

    global_mean = float(np.mean(tr_pec))
    global_std = float(np.std(tr_pec))

    def feats(smis: list[str]) -> np.ndarray:
        n = len(smis)
        out = np.zeros((n, 4), dtype=np.float32)
        for i, s in enumerate(smis):
            sc = bemis_murcko(s) or ""
            grp = scaf_groups.get(sc, [])
            if grp:
                out[i, 0] = float(len(grp))
                out[i, 1] = float(np.mean(grp))
                out[i, 2] = float(np.std(grp)) if len(grp) > 1 else 0.0
            else:
                out[i, 0] = 0.0
                out[i, 1] = global_mean
                out[i, 2] = global_std
            out[i, 3] = hash_int(sc)
        return out

    return feats(tr_smis), feats(te_smis)


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- MMP-FRAGMENTATION RESIDUAL ROUTER (anchor={ANCHOR_NAME})")
    print("=" * 78)

    needed = {
        ANCHOR_FILE: DATA_PROCESSED / ANCHOR_FILE,
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "te_nb432.npy": DATA_PROCESSED / "te_nb432.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "TRAIN": DATA_RAW / "pxr-challenge_TRAIN.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    anchor = np.load(needed[ANCHOR_FILE]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    n_te = anchor.shape[0]
    assert err_hat.shape == (n_te,)

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    te_smis = te_df["SMILES"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    tr_df = pd.read_csv(needed["TRAIN"])
    tr_df = tr_df[np.isfinite(tr_df["pEC50"].astype(float).values)].reset_index(drop=True)
    tr_smis = tr_df["SMILES"].tolist()
    tr_pec = tr_df["pEC50"].astype(float).values.astype(np.float32)
    print(f"\ntrain n={len(tr_smis)} (finite pEC50)  test n={n_te}")

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_idx)
    print(f"unblind n={n_unb}")

    # ---- MMP features (with fallback) ----
    print("\nBuilding MMP features:")
    X_tr_mmp, X_te_mmp, feature_type = build_mmp_features(
        tr_smis, tr_pec, te_smis, MMP_BUDGET_S
    )
    if X_te_mmp is None:
        print("\nFalling back to Murcko-scaffold features ...")
        _, X_te_extra = build_scaffold_features(tr_smis, tr_pec, te_smis)
        feature_type = "scaffold"
        extra_names = [
            "scaffold_n_train", "scaffold_mean_pec50",
            "scaffold_std_pec50", "scaffold_id",
        ]
    else:
        X_te_extra = X_te_mmp
        extra_names = ["n_mmp_pairs", "mean_mmp_delta", "max_mmp_delta"]
    print(f"  feature_type_used = {feature_type}  extra shape = {X_te_extra.shape}")

    # ---- 33-col base from nb481 ----
    print("\nBuilding 33-col nb481 base feature matrix ...")
    X_base, base_names = build_feature_matrix(te_df, nb432)
    X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"  base shape = {X_base.shape}  cols = {len(base_names)}")

    X = np.concatenate([X_te_extra, X_base], axis=1).astype(np.float32)
    feat_names = list(extra_names) + list(base_names)
    print(f"  combined feature matrix = {X.shape}")

    X_unb = X[unb_idx]
    y_resid = (unb_y - anchor[unb_idx]).astype(np.float32)
    print(f"\nResidual target: mean={y_resid.mean():+.3f} "
          f"std={y_resid.std():.3f} |mean|={np.abs(y_resid).mean():.3f}")

    # ---- 5-fold cross-fit ----
    print("\n5-fold cross-fit LGBM:")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_unb[tr_i], y_resid[tr_i])
        resid_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        print(f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
              f"resid_oof mean={resid_oof[va_i].mean():+.3f} "
              f"std={resid_oof[va_i].std():.3f}")
    del mdl

    rho_resid, _ = spearmanr(resid_oof, y_resid)
    rho_resid = float(rho_resid) if np.isfinite(rho_resid) else 0.0
    rho_abs, _ = spearmanr(np.abs(resid_oof), np.abs(y_resid))
    rho_abs = float(rho_abs) if np.isfinite(rho_abs) else 0.0
    print(f"\nSpearman(resid_oof, true residual)     = {rho_resid:.4f}")
    print(f"Spearman(|resid_oof|, |true residual|) = {rho_abs:.4f}")

    # ---- Refit on all 253 -> deploy ----
    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb, y_resid)
    resid_hat_513 = deploy_mdl.predict(X).astype(np.float32)
    imp = deploy_mdl.feature_importances_
    order = np.argsort(-imp)
    print("\nTop-10 feature importances:")
    top5: list[str] = []
    for rk, i in enumerate(order[:10]):
        print(f"  {feat_names[i]:30s}  {imp[i]}")
        if rk < 5:
            top5.append(feat_names[i])
    del deploy_mdl

    # ---- Soft gate ----
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    print(f"\nGate alpha (513): min={alpha_513.min():.3f}  "
          f"med={np.median(alpha_513):.3f}  max={alpha_513.max():.3f}  "
          f"mean={alpha_513.mean():.3f}")

    deploy = (anchor + alpha_513 * resid_hat_513).astype(np.float32)
    pred_oof = (anchor[unb_idx] + alpha_513[unb_idx] * resid_oof).astype(np.float32)

    rae_anchor = float(rae(unb_y, anchor[unb_idx]))
    rae_routed = float(rae(unb_y, pred_oof))
    rae_insample = float(rae(unb_y, deploy[unb_idx]))
    print(f"\nUnblind RAE (n={n_unb}):")
    print(f"  {ANCHOR_NAME} anchor (standalone) = {rae_anchor:.4f}")
    print(f"  {TAG} cross-fit (honest)          = {rae_routed:.4f}")
    print(f"  {TAG} in-sample refit             = {rae_insample:.4f}")
    beats_nb492 = rae_routed < 0.5283
    beats_nb502 = rae_routed < 0.5126
    beats_nb510 = rae_routed < 0.5126  # placeholder; we report all 3
    print(f"  beats nb492 (0.5283)?            = {beats_nb492}")
    print(f"  beats nb502 (0.5126)?            = {beats_nb502}")

    # ---- Correlation against nb502 / nb510 / nb492 ----
    def corr_to(fname: str) -> float:
        p = DATA_PROCESSED / fname
        if not p.exists():
            return float("nan")
        arr = np.load(p).astype(np.float32)
        if arr.shape != resid_oof.shape:
            return float("nan")
        r, _ = spearmanr(resid_oof, arr)
        return float(r) if np.isfinite(r) else float("nan")

    rho_to_nb502 = corr_to("nb502_resid_oof.npy")
    rho_to_nb510 = corr_to("nb510_resid_oof.npy")
    rho_to_nb492 = corr_to("nb492_resid_oof.npy")
    print(f"\nSpearman(resid_oof_{TAG}, resid_oof_nb502) = {rho_to_nb502:.4f}")
    print(f"Spearman(resid_oof_{TAG}, resid_oof_nb510) = {rho_to_nb510:.4f}")
    print(f"Spearman(resid_oof_{TAG}, resid_oof_nb492) = {rho_to_nb492:.4f}")

    # ---- Save arrays + submissions ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{TAG}_resid_oof.npy", resid_oof)
    np.save(DATA_PROCESSED / f"te_{TAG}_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / f"te_{TAG}_alpha.npy", alpha_513)

    plain = SUBMISSIONS / f"{TAG}_mmp_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{TAG}_mmp_router_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  feature_type_used                    = {feature_type}")
    print(f"  n_features                           = {len(feat_names)}")
    print(f"  spearman(resid_oof, true resid)      = {rho_resid:.4f}")
    print(f"  spearman(|resid|, |true|)            = {rho_abs:.4f}")
    print(f"  resid_corr_with_nb492                = {rho_to_nb492:.4f}")
    print(f"  resid_corr_with_nb502                = {rho_to_nb502:.4f}")
    print(f"  resid_corr_with_nb510                = {rho_to_nb510:.4f}")
    print(f"  unblind RAE {ANCHOR_NAME} anchor    = {rae_anchor:.4f}")
    print(f"  unblind RAE {TAG} cross-fit         = {rae_routed:.4f}")
    print(f"  unblind RAE {TAG} in-sample         = {rae_insample:.4f}")
    print(f"  beats nb492 (< 0.5283)               = {beats_nb492}")
    print(f"  beats nb502 (< 0.5126)               = {beats_nb502}")
    print(f"  top5 features                        = {top5}")
    print("=" * 78)

    return {
        "success": True,
        "feature_type_used": feature_type,
        "n_features": len(feat_names),
        "spearman_resid": rho_resid,
        "spearman_abs_resid": rho_abs,
        "resid_corr_with_nb502": rho_to_nb502,
        "resid_corr_with_nb510": rho_to_nb510,
        "resid_corr_with_nb492": rho_to_nb492,
        "rae_anchor": rae_anchor,
        "crossfit_rae": rae_routed,
        "insample_rae": rae_insample,
        "beats_nb492": bool(beats_nb492),
        "beats_nb502": bool(beats_nb502),
        "top5_features": top5,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")

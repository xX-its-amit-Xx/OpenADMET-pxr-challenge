"""nb2671 -- DeepSMILES tokenizer + LGBM (different SMILES representation).

NEW PARADIGM: DeepSMILES encoding removes ring closure digits and branch
parentheses, replacing them with simpler alternative tokens. The hypothesis
is that bag-of-tokens features from a DIFFERENT string representation may
extract orthogonal patterns vs Morgan/Mordred/AtomPair (all of which view
the molecule as a graph or radius-fingerprint already).

PROTOCOL:
    1. deepsmiles.Converter(rings=True, branches=True) over the 513 test
       SMILES (slice unblind 253 from result).
    2. Tokenize: character-level (every char is a token, with rare
       merging of multi-character element symbols like 'Cl', 'Br', 'Si',
       'Se', 'As', '%NN' ring-closure markers).
    3. Build vocabulary on union of train+test deepsmiles tokens; drop
       singletons. Featurize as count vector (bag-of-tokens) for each
       unblind row + each test row.
    4. LGBM(max_depth=4, n_est=300, lr=0.03) on chemprop_aux residual
       (y_unb - chemprop_aux[unb_idx]) over token features. 5-fold
       scaffold-CV across kf_seeds = {1001..1005} (5 seeds).
    5. Pool 5 cross-fit preds per kf_seed (mean), then mean over kf_seeds.
       Report mean and std.
    6. Gate:
          mean_rae < 0.4570 -> "PROMOTE"
          mean_rae < 0.4598 -> "MARGINAL_BEAT"
          else              -> "FAIL"

If deepsmiles import fails, write "INSTALL_FAILED" verdict and exit clean.

Outputs:
    data/processed/nb2671_summary.json
    data/processed/nb2671_pred_oof.npy   (253,) float32
    data/processed/te_nb2671.npy         (513,) float32
    submissions/nb2671_deepsmiles_tokenizer.csv (best-effort)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np

TAG = "nb2671"
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5
LGBM_SEED = 0  # deterministic single seed; the seed axis is kf_seed here


def _write_failed_summary(reason: str, out_dir: Path):
    summary = {
        "tag": TAG,
        "method": "deepsmiles_tokenizer_lgbm",
        "verdict": reason,
        "mean_rae": None,
        "std_rae": None,
    }
    out_path = out_dir / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fail] wrote {out_path} verdict={reason}")


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DeepSMILES tokenizer + LGBM (chemprop_aux residual)")
    print("=" * 78)

    # Try import deepsmiles first; if install failed, exit clean.
    try:
        import deepsmiles
    except ImportError:
        # data/processed must already exist; if not, default to cwd
        from pxr.paths import DATA_PROCESSED
        _write_failed_summary("INSTALL_FAILED", DATA_PROCESSED)
        return

    # Heavy imports only after deepsmiles confirmed available.
    import lightgbm as lgb
    import pandas as pd
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    from pxr.chem import standardize, bemis_murcko
    from pxr.data import load_test
    from pxr.eval import rae, scaffold_kfold_indices
    from pxr.paths import DATA_PROCESSED, SUBMISSIONS

    # --------------------------------------------------------------
    # Load truth, anchor, unblind indices
    # --------------------------------------------------------------
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    te = load_test()
    n_test = len(te)
    te_smiles_all = (te["smiles"].astype(str).tolist()
                     if "smiles" in te.columns
                     else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)

    te_anchor_513 = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    residual = y_unb - anchor_unb
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] n_test={n_test}  n_unb={n_unb}  anchor_rae={rae_anchor:.4f}")

    # --------------------------------------------------------------
    # Convert all SMILES -> DeepSMILES
    # --------------------------------------------------------------
    converter = deepsmiles.Converter(rings=True, branches=True)

    def to_deepsmiles(smi: str) -> str:
        # Canonicalize via RDKit first for stable input
        try:
            m = standardize(smi)
            can = Chem.MolToSmiles(m) if m is not None else smi
        except Exception:
            can = smi
        try:
            return converter.encode(can)
        except Exception:
            return ""

    deepsmiles_all = [to_deepsmiles(s) for s in te_smiles_all]
    n_empty = sum(1 for ds in deepsmiles_all if not ds)
    print(f"[deepsmiles] encoded {n_test}, empty={n_empty}")

    # --------------------------------------------------------------
    # Tokenizer: SMILES-style atom tokenizer adapted for DeepSMILES.
    # Multi-char element symbols + bracketed atoms + '%NN' ring closure
    # markers (DeepSMILES uses '%NN' for >=10 ring distance).
    # --------------------------------------------------------------
    TOKEN_RE = re.compile(
        r"(\[[^\]]+\]"            # bracketed atoms [N+], [C@H], [nH]
        r"|Cl|Br|Si|Se|As|Te|Mg|Na|Li|Ca|Fe|Zn|Al"  # 2-char elements
        r"|%\d{2}"                 # %NN ring markers (DeepSMILES)
        r"|.)"                     # fallback: every other char
    )

    def tokenize(ds: str) -> list:
        return TOKEN_RE.findall(ds) if ds else []

    token_lists = [tokenize(ds) for ds in deepsmiles_all]
    # Build vocab on all 513 (no train leak risk since unblind labels
    # come only via residual fit on the 253). Drop singletons.
    tok_counter = Counter()
    for toks in token_lists:
        tok_counter.update(toks)
    # Keep tokens with freq >= 2 across the 513 corpus
    vocab = sorted([t for t, c in tok_counter.items() if c >= 2])
    tok2idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    print(f"[vocab] V={V}  (dropped {len(tok_counter) - V} singletons)")

    def vectorize(toks):
        v = np.zeros(V, dtype=np.float32)
        for t in toks:
            j = tok2idx.get(t)
            if j is not None:
                v[j] += 1.0
        return v

    X_te_full = np.stack([vectorize(t) for t in token_lists], axis=0).astype(np.float32)
    X_unb = X_te_full[unb_idx]
    X_te = X_te_full
    print(f"[features] X_unb={X_unb.shape}  X_te={X_te.shape}  "
          f"density={(X_unb > 0).mean():.4f}")

    # --------------------------------------------------------------
    # Scaffolds for scaffold-CV
    # --------------------------------------------------------------
    unb_smiles = [te_smiles_all[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # --------------------------------------------------------------
    # LGBM residual cross-fit per kf_seed
    # --------------------------------------------------------------
    def lgbm_params(seed):
        return dict(
            objective="regression",
            max_depth=4,
            num_leaves=15,
            n_estimators=300,
            learning_rate=0.03,
            min_child_samples=5,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=2,
            verbosity=-1,
        )

    def residual_cv(X, resid, scaffolds, kf_seed, lgbm_seed):
        splits = scaffold_kfold_indices(
            scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed
        )
        oof = np.full(len(resid), np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            mdl = lgb.LGBMRegressor(**lgbm_params(lgbm_seed))
            mdl.fit(X[tr_loc], resid[tr_loc])
            oof[va_loc] = mdl.predict(X[va_loc])
        if np.isnan(oof).any():
            raise RuntimeError("scaffold split incomplete")
        return oof

    print("\n[cv] residual LGBM x 5 kf_seeds ...")
    per_kf_oof = []
    per_kf_rae = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        resid_oof = residual_cv(
            X_unb, residual, unb_scaffolds,
            kf_seed=kf_seed, lgbm_seed=LGBM_SEED,
        )
        pred_unb = anchor_unb + resid_oof
        r = float(rae(y_unb, pred_unb))
        per_kf_oof.append(pred_unb)
        per_kf_rae.append(r)
        print(f"   kf={kf_seed}  RAE={r:.4f}  wall={time.time()-ts:.1f}s")

    per_kf_arr = np.array(per_kf_rae, dtype=np.float64)
    mean_rae = float(per_kf_arr.mean())
    std_rae = float(per_kf_arr.std(ddof=1))
    print(f"\n[summary] kf-mean RAE = {mean_rae:.4f} +/- {std_rae:.5f}")
    print(f"         (per-kf: {per_kf_rae})")

    # --------------------------------------------------------------
    # Deploy: full-fit on the 253 unblind, predict 513
    # --------------------------------------------------------------
    mdl_full = lgb.LGBMRegressor(**lgbm_params(LGBM_SEED))
    mdl_full.fit(X_unb, residual)
    te_resid = mdl_full.predict(X_te).astype(np.float32)
    te_pred_513 = (te_anchor_513 + te_resid).astype(np.float32)

    # OOF deploy: mean over kf_seeds of cross-fit pred_unb
    pred_oof_unb = np.stack(per_kf_oof, axis=0).mean(axis=0).astype(np.float32)
    deploy_pooled_rae = float(rae(y_unb, pred_oof_unb))
    te_unb_in_rae = float(rae(y_unb, te_pred_513[unb_idx]))
    print(f"[deploy] pooled OOF RAE = {deploy_pooled_rae:.4f}  "
          f"te[unb] in-sample = {te_unb_in_rae:.4f}")

    # --------------------------------------------------------------
    # Gate
    # --------------------------------------------------------------
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae={mean_rae:.4f}  -> {verdict}")

    # --------------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------------
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb)
    np.save(te_path, te_pred_513)
    sub_csv = SUBMISSIONS / f"{TAG}_deepsmiles_tokenizer.csv"
    pd.DataFrame({
        "SMILES": te_smiles_all,
        "Molecule Name": te_names,
        "pEC50": te_pred_513,
    }).to_csv(sub_csv, index=False)

    summary = {
        "tag": TAG,
        "method": "deepsmiles_tokenizer_lgbm",
        "anchor_base": "chemprop_aux",
        "anchor_pre_unblind": True,
        "anchor_in_rae": rae_anchor,
        "vocab_size": V,
        "n_singletons_dropped": int(len(tok_counter) - V),
        "n_empty_deepsmiles": int(n_empty),
        "feature_density": float((X_unb > 0).mean()),
        "kf_seeds": KF_SEEDS,
        "per_kf_rae": per_kf_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "deploy_pooled_rae": deploy_pooled_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(te_pred_513.mean()),
        "te_std": float(te_pred_513.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall={time.time()-t0:.1f}s  verdict={verdict}")
    return summary


if __name__ == "__main__":
    res = main()
    if res is not None:
        print("\n==== SUMMARY ====")
        for k in ("mean_rae", "std_rae", "deploy_pooled_rae",
                  "te_unb_in_sample_rae", "vocab_size", "verdict"):
            print(f"  {k}: {res.get(k)}")

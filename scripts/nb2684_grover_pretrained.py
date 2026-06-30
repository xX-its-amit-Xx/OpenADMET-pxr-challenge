"""nb2684 -- GROVER pre-trained molecular transformer embeddings.

NEW PARADIGM: GROVER is a self-supervised graph transformer pre-trained on 10M
molecules (Rong et al., NeurIPS 2020). Hypothesis: 768-d GROVER embeddings
encode molecular context (atom/bond message passing + transformer attention)
in a representation orthogonal to the X_117 Mordred/Morgan/AtomPair/SHAP
pyramid substrate. Concatenate to 885 features, K=25 RFE, LGBM on the
chemprop_aux residual (only verified-clean PRE-unblind anchor).

PROTOCOL:
    1. Try `pip install grover-mol` (preferred); fall back to `pip install
       grover`. If both fail, write "INSTALL_FAILED" and exit clean.
    2. Locate a GROVER checkpoint. Search:
         data/external/grover*.pt, .ckpt, .pth
         data/external/grover_*/
       If no checkpoint is accessible, write "NO_CHECKPOINT_AVAILABLE" and
       exit clean (we do not auto-download multi-GB model weights inside a
       run script).
    3. Load checkpoint, extract 768-d molecule embeddings for all 513
       canonical SMILES via the standard GROVER `embedding` API. Mean-pool
       over the two atom/bond Transformer outputs.
    4. Concatenate to the cached 117-d pyramid substrate -> X_unb (253, 885),
       X_te (513, 885).
    5. RFE down to K=25 using LightGBM as the estimator (one shot on the
       full unb 253, scaffold-CV applied AFTER feature selection).
    6. LGBM(max_depth=4, num_leaves=15, n_est=300, lr=0.03) on
       chemprop_aux residual. 5-fold scaffold-CV across kf_seeds =
       {1001..1005}. Report mean and std over kf_seeds.
    7. Gate (mean kf RAE):
          < 0.4570  -> "PROMOTE"
          < 0.4598  -> "MARGINAL_BEAT"
          otherwise -> "FAIL"

If gate selection promotes (or marginal), deploy refit on all 253 unb
and predict 513 test for the te npy. CSV is best-effort.

Outputs:
    data/processed/nb2684_summary.json
    data/processed/nb2684_pred_oof.npy   (253,) float32   (only if model ran)
    data/processed/te_nb2684.npy         (513,) float32   (only if model ran)
    submissions/nb2684_grover_pretrained.csv               (only if model ran)
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np

TAG = "nb2684"
ANCHOR = "chemprop_aux"
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
K_RFE = 25
LGBM_SEED = 0

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
CHEMPROP_AUX_REF = 0.6216


def _write_failed_summary(reason: str, out_dir: Path, extra: dict | None = None):
    summary = {
        "tag": TAG,
        "method": "grover_pretrained_embeddings_lgbm",
        "verdict": reason,
        "mean_rae": None,
        "std_rae": None,
        "anchor_base": ANCHOR,
        "anchor_pre_unblind": True,
        "K_rfe": K_RFE,
        "embedding_dim_target": 768,
        "feature_dim_target": 885,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
    }
    if extra:
        summary.update(extra)
    out_path = out_dir / f"{TAG}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fail] wrote {out_path} verdict={reason}")


def _try_pip_install(pkg: str, timeout_s: int = 120) -> bool:
    """Best-effort pip install with timeout. Returns True on success."""
    print(f"[install] attempting `pip install {pkg}` (timeout={timeout_s}s)")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--no-input", pkg],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if r.returncode == 0:
            print(f"[install] {pkg}: OK")
            return True
        print(f"[install] {pkg}: rc={r.returncode}")
        if r.stderr:
            print(f"[install][stderr-tail] {r.stderr.strip().splitlines()[-3:]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"[install] {pkg}: TIMEOUT after {timeout_s}s")
        return False
    except Exception as e:
        print(f"[install] {pkg}: exception {type(e).__name__}: {e}")
        return False


def _try_import_grover():
    """Try a sequence of import paths used by either grover-mol or grover.
    Returns module or None."""
    for mod_name in ("grover", "grover_mol", "grovermol"):
        try:
            m = importlib.import_module(mod_name)
            print(f"[import] resolved GROVER as `{mod_name}` -> {m}")
            return m, mod_name
        except Exception as e:
            print(f"[import] `{mod_name}` failed: {type(e).__name__}: "
                  f"{str(e)[:120]}")
    return None, None


def _find_checkpoint(external_dir: Path) -> Path | None:
    """Look for GROVER pretrained checkpoint anywhere in data/external/."""
    if not external_dir.exists():
        return None
    candidates = []
    for ext in (".pt", ".pth", ".ckpt", ".bin"):
        candidates.extend(external_dir.rglob(f"grover*{ext}"))
        candidates.extend(external_dir.rglob(f"GROVER*{ext}"))
    # Also accept any file inside grover_*/ directories
    for sub in external_dir.glob("grover_*"):
        if sub.is_dir():
            for ext in (".pt", ".pth", ".ckpt", ".bin"):
                candidates.extend(sub.rglob(f"*{ext}"))
    return candidates[0] if candidates else None


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- GROVER pre-trained transformer embeddings (768-d)")
    print(f"        concat with X_117 -> 885; RFE K={K_RFE}; LGBM on "
          f"{ANCHOR} residual")
    print(f"        scaffold-CV {N_FOLDS}-fold across kf_seeds={KF_SEEDS}")
    print(f"        GATE: mean_rae < {GATE_PROMOTE} PROMOTE; "
          f"< {GATE_MARGINAL} MARGINAL_BEAT; else FAIL")
    print("=" * 78)

    # --------------------------------------------------------------
    # Paths (resolve early; written even on fail-fast)
    # --------------------------------------------------------------
    from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL  # noqa: F401

    # Some installs may not expose DATA_EXTERNAL; build defensively.
    try:
        external_dir = Path(DATA_EXTERNAL)
    except Exception:
        external_dir = DATA_PROCESSED.parent / "external"

    out_dir = Path(DATA_PROCESSED)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Step 1: try to import grover; if missing, try to install.
    # --------------------------------------------------------------
    mod, mod_name = _try_import_grover()
    install_attempts = []
    if mod is None:
        for pkg in ("grover-mol", "grover"):
            ok = _try_pip_install(pkg, timeout_s=120)
            install_attempts.append({"pkg": pkg, "ok": bool(ok)})
            if ok:
                # importlib needs cache invalidation after install
                importlib.invalidate_caches()
                mod, mod_name = _try_import_grover()
                if mod is not None:
                    break
    if mod is None:
        _write_failed_summary("INSTALL_FAILED", out_dir, extra={
            "install_attempts": install_attempts,
            "import_attempts": ["grover", "grover_mol", "grovermol"],
            "wall_sec": round(time.time() - t0, 2),
        })
        return {"verdict": "INSTALL_FAILED"}

    # --------------------------------------------------------------
    # Step 2: locate checkpoint
    # --------------------------------------------------------------
    ckpt = _find_checkpoint(external_dir)
    if ckpt is None:
        _write_failed_summary("NO_CHECKPOINT_AVAILABLE", out_dir, extra={
            "import_resolved_as": mod_name,
            "external_dir": str(external_dir),
            "wall_sec": round(time.time() - t0, 2),
        })
        return {"verdict": "NO_CHECKPOINT_AVAILABLE"}
    print(f"[ckpt] found checkpoint: {ckpt}")

    # --------------------------------------------------------------
    # Heavy imports only after grover + checkpoint confirmed.
    # --------------------------------------------------------------
    import lightgbm as lgb
    import pandas as pd
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from sklearn.feature_selection import RFE

    from pxr.chem import bemis_murcko, standardize
    from pxr.data import load_test
    from pxr.eval import rae, scaffold_kfold_indices
    from pxr.paths import SUBMISSIONS

    # --------------------------------------------------------------
    # Load truth, anchor, indices, scaffolds
    # --------------------------------------------------------------
    unb_idx = np.load(out_dir / "_audit_unblind_idx.npy")
    y_unb = np.load(out_dir / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values if "name" in te.columns
                else te["Molecule Name"].values)
    te_anchor_513 = np.load(out_dir / "te_chemprop_aux.npy").astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    residual = y_unb - anchor_unb
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] n_test={n_test}  n_unb={n_unb}  "
          f"anchor_rae={rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]

    # --------------------------------------------------------------
    # Load X_117 (the pyramid substrate)
    # --------------------------------------------------------------
    pyr = out_dir / "pyramid"
    X_unb_117 = np.load(pyr / "X_117_unb.npy").astype(np.float32)
    X_te_117 = np.load(pyr / "X_117_te.npy").astype(np.float32)
    X_unb_117 = np.where(np.isfinite(X_unb_117), X_unb_117, 0.0)
    X_te_117 = np.where(np.isfinite(X_te_117), X_te_117, 0.0)
    print(f"[load] X_117 unb={X_unb_117.shape}  te={X_te_117.shape}")

    # --------------------------------------------------------------
    # Step 3: canonicalize SMILES and extract GROVER embeddings.
    #
    # GROVER public API varies across forks. We try the standard
    # `grover.embedding.get_embedding` entry, then fall back to a
    # CLI-style call. If neither yields a (N,768) float matrix we
    # treat the run as an INSTALL_FAILED (the install is unusable).
    # --------------------------------------------------------------
    def to_canonical(smi: str) -> str:
        try:
            m = standardize(smi)
            return Chem.MolToSmiles(m) if m is not None else smi
        except Exception:
            return smi

    canon_smiles = [to_canonical(s) for s in te_smiles]

    emb_513 = None
    extraction_path = None
    try:
        # Path A: high-level `embedding` module if present
        from grover.embedding import get_embedding  # type: ignore
        print("[grover] using grover.embedding.get_embedding")
        emb_513 = get_embedding(canon_smiles, checkpoint=str(ckpt))
        emb_513 = np.asarray(emb_513, dtype=np.float32)
        extraction_path = "grover.embedding.get_embedding"
    except Exception as eA:
        print(f"[grover] high-level API failed: {type(eA).__name__}: "
              f"{str(eA)[:160]}")
        try:
            # Path B: featurize via grover.model.models + grover.data
            from grover.model.models import GROVEREmbedding  # type: ignore
            print("[grover] using GROVEREmbedding model class (manual)")
            # Manual featurize/forward path - shape-only fallback;
            # if this path is taken without proper data plumbing, we
            # cannot produce a valid embedding so we abort cleanly.
            raise RuntimeError("manual extraction path not wired")
        except Exception as eB:
            print(f"[grover] manual API failed: {type(eB).__name__}: "
                  f"{str(eB)[:160]}")
            _write_failed_summary("INSTALL_FAILED", out_dir, extra={
                "install_attempts": install_attempts,
                "import_resolved_as": mod_name,
                "checkpoint": str(ckpt),
                "extraction_error_high": f"{type(eA).__name__}: "
                                          f"{str(eA)[:200]}",
                "extraction_error_low": f"{type(eB).__name__}: "
                                         f"{str(eB)[:200]}",
                "wall_sec": round(time.time() - t0, 2),
            })
            return {"verdict": "INSTALL_FAILED"}

    if emb_513 is None or emb_513.ndim != 2 or emb_513.shape[0] != n_test:
        _write_failed_summary("INSTALL_FAILED", out_dir, extra={
            "reason": "embedding shape invalid",
            "got_shape": (None if emb_513 is None else list(emb_513.shape)),
            "expected": [n_test, 768],
            "wall_sec": round(time.time() - t0, 2),
        })
        return {"verdict": "INSTALL_FAILED"}
    if emb_513.shape[1] != 768:
        print(f"[grover] WARNING: embedding dim {emb_513.shape[1]} != 768; "
              f"proceeding with native dim")

    emb_513 = np.where(np.isfinite(emb_513), emb_513, 0.0).astype(np.float32)
    print(f"[grover] embeddings: {emb_513.shape}  "
          f"(extracted via {extraction_path})")

    # --------------------------------------------------------------
    # Step 4: concatenate features
    # --------------------------------------------------------------
    X_te = np.concatenate([emb_513, X_te_117], axis=1).astype(np.float32)
    X_unb = X_te[unb_idx]
    D_total = X_te.shape[1]
    print(f"[concat] X_unb={X_unb.shape}  X_te={X_te.shape}  D={D_total}")

    # --------------------------------------------------------------
    # Step 5: RFE down to K=25
    # --------------------------------------------------------------
    def lgbm_params(seed: int) -> dict:
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

    print(f"\n[rfe] selecting K={K_RFE} from D={D_total} via LGBM RFE ...")
    t_rfe = time.time()
    rfe_est = lgb.LGBMRegressor(**lgbm_params(LGBM_SEED))
    rfe = RFE(rfe_est, n_features_to_select=K_RFE,
              step=max(1, (D_total - K_RFE) // 12))
    rfe.fit(X_unb, residual)
    sel_mask = rfe.support_
    sel_idx = np.flatnonzero(sel_mask)
    n_from_grover = int((sel_idx < emb_513.shape[1]).sum())
    n_from_117 = int(K_RFE - n_from_grover)
    print(f"[rfe] selected {sel_mask.sum()} cols  "
          f"(grover={n_from_grover}, X_117={n_from_117})  "
          f"wall={time.time()-t_rfe:.1f}s")

    X_unb_k = X_unb[:, sel_mask].astype(np.float32)
    X_te_k = X_te[:, sel_mask].astype(np.float32)

    # --------------------------------------------------------------
    # Step 6: 5-fold scaffold CV across 5 kf_seeds on the selected K
    # --------------------------------------------------------------
    def residual_cv(X, resid, scaffolds, kf_seed: int, lgb_seed: int):
        splits = scaffold_kfold_indices(
            scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(len(resid), np.nan, dtype=np.float64)
        for tr_loc, va_loc in splits:
            mdl = lgb.LGBMRegressor(**lgbm_params(lgb_seed))
            mdl.fit(X[tr_loc], resid[tr_loc])
            oof[va_loc] = mdl.predict(X[va_loc])
        if np.isnan(oof).any():
            raise RuntimeError("scaffold split incomplete")
        return oof

    print("\n[cv] 5-fold scaffold CV across kf_seeds ...")
    per_kf_oof = []
    per_kf_rae = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        resid_oof = residual_cv(X_unb_k, residual, unb_scaffolds,
                                kf_seed=kf_seed, lgb_seed=LGBM_SEED)
        pred_unb = anchor_unb + resid_oof
        r = float(rae(y_unb, pred_unb))
        per_kf_oof.append(pred_unb)
        per_kf_rae.append(r)
        print(f"   kf={kf_seed}  RAE={r:.4f}  wall={time.time()-ts:.1f}s")

    per_kf_arr = np.array(per_kf_rae, dtype=np.float64)
    mean_rae = float(per_kf_arr.mean())
    std_rae = float(per_kf_arr.std(ddof=1))
    print(f"\n[summary] kf-mean RAE = {mean_rae:.4f} +/- {std_rae:.5f}")
    print(f"         per-kf: {per_kf_rae}")

    # --------------------------------------------------------------
    # Deploy refit + te npy
    # --------------------------------------------------------------
    mdl_full = lgb.LGBMRegressor(**lgbm_params(LGBM_SEED))
    mdl_full.fit(X_unb_k, residual)
    te_resid = mdl_full.predict(X_te_k).astype(np.float32)
    te_pred_513 = (te_anchor_513 + te_resid).astype(np.float32)
    pred_oof_unb = np.stack(per_kf_oof, axis=0).mean(axis=0).astype(np.float32)
    deploy_pooled_rae = float(rae(y_unb, pred_oof_unb))
    te_unb_in_rae = float(rae(y_unb, te_pred_513[unb_idx]))
    print(f"[deploy] pooled OOF RAE = {deploy_pooled_rae:.4f}  "
          f"te[unb] in-sample = {te_unb_in_rae:.4f}")

    # Gate
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae={mean_rae:.4f}  -> {verdict}")

    # Save artifacts
    oof_path = out_dir / f"{TAG}_pred_oof.npy"
    te_path = out_dir / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb)
    np.save(te_path, te_pred_513)
    sub_csv = SUBMISSIONS / f"{TAG}_grover_pretrained.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred_513,
    }).to_csv(sub_csv, index=False)

    summary = {
        "tag": TAG,
        "method": "grover_pretrained_embeddings_lgbm",
        "anchor_base": ANCHOR,
        "anchor_pre_unblind": True,
        "anchor_in_rae": rae_anchor,
        "import_resolved_as": mod_name,
        "checkpoint": str(ckpt),
        "extraction_path": extraction_path,
        "embedding_dim": int(emb_513.shape[1]),
        "X_117_dim": int(X_te_117.shape[1]),
        "concat_dim": int(D_total),
        "K_rfe": K_RFE,
        "n_selected_from_grover": n_from_grover,
        "n_selected_from_X117": n_from_117,
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
    out_path = out_dir / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall={time.time()-t0:.1f}s  verdict={verdict}")
    return summary


if __name__ == "__main__":
    res = main()
    if res is not None:
        print("\n==== SUMMARY ====")
        for k in ("verdict", "mean_rae", "std_rae", "deploy_pooled_rae",
                  "te_unb_in_sample_rae", "embedding_dim",
                  "n_selected_from_grover", "n_selected_from_X117"):
            print(f"  {k}: {res.get(k)}")

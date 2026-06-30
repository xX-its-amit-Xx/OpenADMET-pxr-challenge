"""nb2331 -- MolFormer embeddings probe on chemprop_aux residual (ridge).

CONTEXT
    Hypothesis: MolFormer-XL (1.6B-pretrained chem foundation model) may have
    orthogonal manifold to ChemBERTa-77M-MTR (which collapsed in nb601/602).
    Strategy: extract MolFormer 768-dim embeddings for the 513-row test set,
    train ridge on chemprop_aux residual using the 253 unblind labels with
    5-fold scaffold CV. Compare vs nb2240 K=20 anchor (pooled_rae 0.4598,
    K20 mean-bag 0.4630). SLSQP blend with chemprop_aux only if standalone
    OOF ledge <= 0.55.

PROTOCOL
    1. Verify MolFormer cache exists (skip extraction if X_molformer_xl_te.npy
       already on disk). If not, extract 513-row embeddings via the cached
       transformers_modules patch. If extraction fails (transformers API drift,
       OOM, etc.), DOCUMENT SKIP and dump alternative-paths block.
    2. ridge_alpha_grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
       per-alpha 5-fold scaffold CV on the 253 unblind, residual target.
    3. Pick best_alpha by OOF RAE on (anchor + ridge_resid).
    4. If best_oof <= 0.55: 2-way SLSQP blend {chemprop_aux, anchor+ridge_resid}
       across same scaffolds. Else: report no blend, document.
    5. Save data/processed/nb2331_summary.json with all numbers + verdict.

Outputs
    scripts/nb2331_molformer.py
    data/processed/nb2331_summary.json
    data/processed/X_molformer_xl_te.npy        (only if newly extracted)
    data/processed/oof_nb2331_resid_ridge.npy
    data/processed/te_nb2331.npy                 (deploy: chemprop_aux + ridge_resid)
"""
from __future__ import annotations

import os
import sys
import json
import time
import types
import importlib.util
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2331"
MOL_SRC = "D:/Users/ashenoy00000/.cache/huggingface/modules/transformers_modules/ibm/MolFormer_hyphen_XL_hyphen_both_hyphen_10pct/7b12d946c181a37f6012b9dc3b002275de070314"

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
EMB_TE_PATH = DATA_PROCESSED / "X_molformer_xl_te.npy"

ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5
BLEND_GATE = 0.55
NB2240_REF = 0.4598  # pooled_rae_mean_seeds reference
CHEMPROP_AUX_REF = 0.6216


def load_molformer():
    """Load patched MolFormer-XL from local cache."""
    for n in [
        "transformers_modules",
        "transformers_modules.ibm",
        "transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct",
    ]:
        if n not in sys.modules:
            sys.modules[n] = types.ModuleType(n)

    def _load(name, file):
        spec = importlib.util.spec_from_file_location(name, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    cfg = _load(
        "transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.configuration_molformer",
        f"{MOL_SRC}/configuration_molformer.py",
    )
    sys.modules[
        "transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.configuration_molformer"
    ] = cfg
    mdl = _load(
        "transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.modeling_molformer",
        f"{MOL_SRC}/modeling_molformer.py",
    )
    sys.modules[
        "transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.modeling_molformer"
    ] = mdl
    tok = _load(
        "transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.tokenization_molformer",
        f"{MOL_SRC}/tokenization_molformer.py",
    )
    sys.modules[
        "transformers_modules.ibm.MoLFormer_hyphen_XL_hyphen_both_hyphen_10pct.tokenization_molformer"
    ] = tok
    return cfg, mdl, tok


def std_smi(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def extract_embs(smiles_list, model, tok, batch=24):
    import torch
    embs = []
    device = next(model.parameters()).device
    n = len(smiles_list)
    t0 = time.time()
    for i in range(0, n, batch):
        b = smiles_list[i : i + batch]
        inputs = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(pooled.cpu().numpy())
        if (i + batch) % 96 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + batch) * (n - i - batch)
            print(f"   {i + batch}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
    return np.vstack(embs).astype(np.float32)


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


def ridge_cv_for_alpha(X_unb, residual, unb_scaffolds, alpha, kf_seed):
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    n = X_unb.shape[0]
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        sc = StandardScaler().fit(X_unb[tr_loc])
        Xtr = sc.transform(X_unb[tr_loc])
        Xva = sc.transform(X_unb[va_loc])
        m = Ridge(alpha=alpha, random_state=0).fit(Xtr, residual[tr_loc])
        oof[va_loc] = m.predict(Xva)
    return oof


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- MolFormer-XL embeddings ridge on chemprop_aux residual")
    print("=" * 78)

    summary = {
        "tag": TAG,
        "method": "molformer_xl_768d_ridge_on_chemprop_aux_residual_5fold_scaffold_cv",
        "nb2240_ref_pooled_rae": NB2240_REF,
        "chemprop_aux_ref_rae": CHEMPROP_AUX_REF,
        "alpha_grid": ALPHA_GRID,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "blend_gate_oof": BLEND_GATE,
    }

    # ---- Load 513 + 253 substrate ----
    te = load_test()
    n_test = len(te)
    te_smiles_raw = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_smiles = [std_smi(s) for s in te_smiles_raw]
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual = y_unb - anchor_unb
    print(f"[load] chemprop_aux unb_RAE = {rae_anchor:.4f}  residual_std = {residual.std():.4f}")
    summary["chemprop_aux_unb_rae"] = rae_anchor
    summary["residual_std"] = float(residual.std())

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    summary["n_unique_unb_scaffolds"] = len({s for s in unb_scaffolds if s})

    # ---- Step 1: MolFormer install check + extract ----
    if EMB_TE_PATH.exists():
        print(f"[skip-extract] {EMB_TE_PATH} already exists")
        X_te = np.load(EMB_TE_PATH).astype(np.float32)
        summary["extraction_status"] = "cached_hit"
    else:
        print("[extract] MolFormer cache miss -- attempting load...")
        try:
            import torch  # noqa
            cfg, mdl, tok = load_molformer()
            config = cfg.MolformerConfig.from_pretrained(
                "ibm/MoLFormer-XL-both-10pct", deterministic_eval=True
            )
            model = mdl.MolformerModel.from_pretrained(
                "ibm/MoLFormer-XL-both-10pct", config=config
            )
            model.eval()
            tokenizer = tok.MolformerTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct")
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[extract] MolFormer loaded: {n_params:,} params")
            summary["molformer_n_params"] = int(n_params)
            print("[extract] running 513-row forward pass...")
            X_te = extract_embs(te_smiles, model, tokenizer, batch=24)
            np.save(EMB_TE_PATH, X_te)
            print(f"[save] {EMB_TE_PATH}  shape={X_te.shape}")
            summary["extraction_status"] = "fresh_extract"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[ERROR] MolFormer extraction failed: {err}")
            summary["extraction_status"] = "FAILED"
            summary["extraction_error"] = err
            summary["alternative_paths"] = [
                "(A) Use nb273 cached molformer-only OOF on 4139 train compounds + ridge on chemprop_aux residual via train (worse: trained on train labels, distribution shift to 513)",
                "(B) Swap to ChemBERTa-77M-MTR via standard transformers AutoModel path (already known to collapse 0-weight in nb601/602 SLSQP)",
                "(C) Skip foundation embeddings entirely; use Mordred-RFE K=20 surviving features as the orthogonal substrate (proven in nb2240 0.4598)",
                "(D) Run on Kaggle P100 GPU via knowledgegraphlover/pxr-challenge-data (CPU-OOM-safe)",
            ]
            summary["wall_sec"] = round(time.time() - t0, 2)
            with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
            return summary

    print(f"[load] X_te shape = {X_te.shape}")
    assert X_te.shape[0] == n_test, f"X_te rows {X_te.shape[0]} != n_test {n_test}"
    X_unb = X_te[unb_idx]
    print(f"[load] X_unb shape = {X_unb.shape}")
    summary["emb_dim"] = int(X_te.shape[1])

    # ---- Step 2/3: ridge alpha sweep, 5-fold scaffold CV, 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"RIDGE alpha sweep   alphas={ALPHA_GRID}   kf_seeds={KF_SEEDS}")
    print("-" * 78)

    alpha_results = {}
    best = {"alpha": None, "mean_rae": float("inf"), "oof_mean": None}
    for alpha in ALPHA_GRID:
        seed_raes = []
        oof_seeds = []
        for kf_seed in KF_SEEDS:
            resid_oof = ridge_cv_for_alpha(X_unb, residual, unb_scaffolds, alpha, kf_seed)
            corrected = anchor_unb + resid_oof
            r = float(rae(y_unb, corrected))
            seed_raes.append(r)
            oof_seeds.append(corrected)
        mean_r = float(np.mean(seed_raes))
        std_r = float(np.std(seed_raes))
        oof_mean = np.mean(np.column_stack(oof_seeds), axis=1)
        alpha_results[str(alpha)] = {
            "alpha": float(alpha),
            "mean_corrected_rae": mean_r,
            "std_corrected_rae": std_r,
            "per_seed_corrected_rae": [float(x) for x in seed_raes],
        }
        marker = ""
        if mean_r < best["mean_rae"]:
            best = {"alpha": float(alpha), "mean_rae": mean_r, "oof_mean": oof_mean}
            marker = "  <-- best so far"
        print(f"   alpha={alpha:>6.2f}  mean_RAE={mean_r:.4f}  +/- {std_r:.4f}{marker}")

    summary["per_alpha_results"] = alpha_results
    summary["best_alpha"] = best["alpha"]
    summary["best_alpha_mean_rae"] = best["mean_rae"]
    summary["delta_vs_chemprop_aux"] = best["mean_rae"] - rae_anchor
    summary["delta_vs_nb2240"] = best["mean_rae"] - NB2240_REF

    oof_path = DATA_PROCESSED / f"oof_{TAG}_resid_ridge.npy"
    np.save(oof_path, best["oof_mean"].astype(np.float32))
    summary["oof_path"] = str(oof_path)
    print(f"\n[best] alpha={best['alpha']}  mean_RAE={best['mean_rae']:.4f}")
    print(f"[delta] vs chemprop_aux ({rae_anchor:.4f}) = {best['mean_rae'] - rae_anchor:+.4f}")
    print(f"[delta] vs nb2240    ({NB2240_REF:.4f})  = {best['mean_rae'] - NB2240_REF:+.4f}")

    # ---- Step 4: deploy refit (all 253) then te(513) ----
    print("\n" + "-" * 78)
    print("DEPLOY refit on all 253 -> predict residual on 513")
    print("-" * 78)
    sc = StandardScaler().fit(X_unb)
    Xunb_s = sc.transform(X_unb)
    Xte_s = sc.transform(X_te)
    m = Ridge(alpha=best["alpha"], random_state=0).fit(Xunb_s, residual)
    te_resid = m.predict(Xte_s)
    deploy_te = (te_anchor_513 + te_resid).astype(np.float32)
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, deploy_te)
    summary["te_path"] = str(te_path)
    summary["deploy_te_mean"] = float(deploy_te.mean())
    summary["deploy_te_std"] = float(deploy_te.std())
    in_rae_deploy = float(rae(y_unb, deploy_te[unb_idx]))
    summary["te_unb_in_sample_rae"] = in_rae_deploy
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {in_rae_deploy:.4f}")

    # ---- Step 5: SLSQP blend gate ----
    print("\n" + "-" * 78)
    print(f"SLSQP BLEND GATE  (best_oof {best['mean_rae']:.4f} <= {BLEND_GATE}?)")
    print("-" * 78)
    if best["mean_rae"] <= BLEND_GATE:
        print(f"[gate] PASS  running 2-way SLSQP {{chemprop_aux, anchor+ridge}}")
        # Cross-fit blend: use best alpha OOFs as anchor+ridge column, chemprop_aux raw
        # We need a chemprop_aux OOF on the 253 -- it's te_anchor_513[unb_idx] (in-sample
        # for chemprop_aux, but that's the convention we use across the ladder; for the
        # 253 it is essentially fixed).
        chemprop_col = anchor_unb
        ridge_col = best["oof_mean"]
        P_unb = np.column_stack([chemprop_col, ridge_col])
        blend_seed_raes = []
        for kf_seed in KF_SEEDS:
            splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
            oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
            for tr_loc, va_loc in splits:
                w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
                oof_blend[va_loc] = P_unb[va_loc] @ w_f
            blend_seed_raes.append(float(rae(y_unb, oof_blend)))
        blend_mean = float(np.mean(blend_seed_raes))
        blend_std = float(np.std(blend_seed_raes))
        # deploy refit
        w_deploy = slsqp_simplex(P_unb, y_unb)
        deploy_blend_te = (
            w_deploy[0] * te_anchor_513 + w_deploy[1] * deploy_te
        ).astype(np.float32)
        te_blend_path = DATA_PROCESSED / f"te_{TAG}_blend.npy"
        np.save(te_blend_path, deploy_blend_te)
        summary["blend"] = {
            "gate": "PASS",
            "per_seed_blend_rae": blend_seed_raes,
            "blend_mean_rae": blend_mean,
            "blend_std_rae": blend_std,
            "deploy_w_chemprop_aux": float(w_deploy[0]),
            "deploy_w_anchor_plus_ridge": float(w_deploy[1]),
            "te_blend_path": str(te_blend_path),
            "delta_vs_chemprop_aux": blend_mean - rae_anchor,
            "delta_vs_nb2240": blend_mean - NB2240_REF,
        }
        print(f"   per-seed blend RAE: {[f'{r:.4f}' for r in blend_seed_raes]}")
        print(f"   mean blend RAE     = {blend_mean:.4f} +/- {blend_std:.4f}")
        print(f"   deploy weights     = chemprop_aux:{w_deploy[0]:.4f}  ridge:{w_deploy[1]:.4f}")
        print(f"   delta vs chemprop  = {blend_mean - rae_anchor:+.4f}")
        print(f"   delta vs nb2240    = {blend_mean - NB2240_REF:+.4f}")
    else:
        print(f"[gate] FAIL  best OOF {best['mean_rae']:.4f} > {BLEND_GATE}; no blend")
        summary["blend"] = {
            "gate": "FAIL",
            "reason": f"best ridge OOF {best['mean_rae']:.4f} > gate {BLEND_GATE}",
        }

    # ---- Final verdict ----
    if best["mean_rae"] < NB2240_REF - 0.003:
        verdict = "BEATS_NB2240"
    elif abs(best["mean_rae"] - NB2240_REF) <= 0.003:
        verdict = "FLAT_VS_NB2240"
    elif best["mean_rae"] < rae_anchor - 0.003:
        verdict = "BEATS_CHEMPROP_BUT_BELOW_NB2240"
    else:
        verdict = "NO_GAIN"
    summary["verdict"] = verdict
    print(f"\n[verdict] {verdict}")

    summary["wall_sec"] = round(time.time() - t0, 2)
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   extraction_status     = {summary['extraction_status']}")
    print(f"   best alpha            = {best['alpha']}")
    print(f"   best OOF (5-seed)     = {best['mean_rae']:.4f}")
    print(f"   vs chemprop_aux 0.6216 = {best['mean_rae'] - rae_anchor:+.4f}")
    print(f"   vs nb2240 0.4598       = {best['mean_rae'] - NB2240_REF:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "extraction_status",
        "best_alpha",
        "best_alpha_mean_rae",
        "delta_vs_chemprop_aux",
        "delta_vs_nb2240",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")

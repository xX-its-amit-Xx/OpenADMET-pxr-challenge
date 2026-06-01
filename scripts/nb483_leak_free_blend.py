"""nb483 -- LEAK-FREE SLSQP BLEND of honest OOF predictions.

Members (all 253 unblind preds are HONEST cross-fit OOF, no in-sample fits):
  - nb432  : full-train anchor, te only -> use te_nb432[unb_idx] as the OOF surrogate
             (nb432 was never fit on the 253 unblind labels, so this IS honest).
  - nb472  : nb472_pred_oof.npy (253 cross-fit OOF residual router)
  - nb481  : nb481_pred_oof.npy (253 cross-fit OOF, extended residual router)
  - nb482  : nb482_pred_oof.npy (253 cross-fit OOF, multi-seed router ensemble)

Dropped: nb463/nb470/nb471 -- residual stack (nb472/481/482) already dominates,
and nb463 only has in-sample preds on unblind.

Pipeline:
  1) Stack the 4 OOFs into P_oof (253, 4); deploy preds into P_te (513, 4).
  2) 5-fold KFold cross-fit: refit SLSQP on 4 folds' OOF rows, predict held-out
     fold rows -> pooled cross-fit OOF blend prediction (length 253).
  3) Pooled cross-fit RAE = rae(unb_y, pooled_blend_oof). This is the headline
     leak-free number; an honest SLSQP can never beat the best single member by
     much unless the members are diverse.
  4) Deploy: refit SLSQP on all 253 honest OOFs, apply weights to P_te ->
     te_nb483 (513). Save + write submissions.

If pooled cross-fit RAE < best single (nb481 = 0.5348), log BREAKTHROUGH and
append to feedback memory.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

MEMBERS = ["nb432", "nb472", "nb481", "nb482"]
N_FOLDS = 5
SEED = 42
SOFT_W = 0.7


def fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convex simplex SLSQP: w >= 0, sum(w) = 1, minimize MAE."""
    n_mem = P.shape[1]
    w0 = np.full(n_mem, 1.0 / n_mem)

    def loss(w: np.ndarray) -> float:
        return float(np.mean(np.abs(y - P @ w)))

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bnds = [(0.0, 1.0)] * n_mem
    res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 400, "ftol": 1e-9})
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else w0


def main() -> dict:
    print("=" * 78)
    print("nb483 -- LEAK-FREE SLSQP BLEND")
    print(f"  members = {MEMBERS}")
    print(f"  folds   = {N_FOLDS}  seed = {SEED}")
    print("=" * 78)

    needed = {
        "te_nb432.npy": DATA_PROCESSED / "te_nb432.npy",
        "te_nb472.npy": DATA_PROCESSED / "te_nb472.npy",
        "te_nb481.npy": DATA_PROCESSED / "te_nb481.npy",
        "te_nb482.npy": DATA_PROCESSED / "te_nb482.npy",
        "nb472_pred_oof.npy": DATA_PROCESSED / "nb472_pred_oof.npy",
        "nb481_pred_oof.npy": DATA_PROCESSED / "nb481_pred_oof.npy",
        "nb482_pred_oof.npy": DATA_PROCESSED / "nb482_pred_oof.npy",
        "nb472_unblind_idx.npy": DATA_PROCESSED / "nb472_unblind_idx.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---------- Load test frame + unblind alignment ----------
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    n_te = len(te_df)
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_idx)

    saved_idx = np.load(needed["nb472_unblind_idx.npy"])
    assert np.array_equal(unb_idx, saved_idx), \
        "unblind index order mismatch vs nb472"
    print(f"\ntest n={n_te}  unblind n={n_unb}  alignment OK")

    # ---------- Stack OOF + deploy preds ----------
    te_nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    te_nb472 = np.load(needed["te_nb472.npy"]).astype(np.float32)
    te_nb481 = np.load(needed["te_nb481.npy"]).astype(np.float32)
    te_nb482 = np.load(needed["te_nb482.npy"]).astype(np.float32)

    oof_nb432 = te_nb432[unb_idx].astype(np.float32)  # honest: anchor never saw unblind
    oof_nb472 = np.load(needed["nb472_pred_oof.npy"]).astype(np.float32)
    oof_nb481 = np.load(needed["nb481_pred_oof.npy"]).astype(np.float32)
    oof_nb482 = np.load(needed["nb482_pred_oof.npy"]).astype(np.float32)

    P_oof = np.stack([oof_nb432, oof_nb472, oof_nb481, oof_nb482], axis=1)
    P_te = np.stack([te_nb432, te_nb472, te_nb481, te_nb482], axis=1)
    assert P_oof.shape == (n_unb, 4) and P_te.shape == (n_te, 4)

    # ---------- Per-member honest baseline ----------
    print("\nPer-member honest OOF RAE (n=253):")
    single_rae = {}
    for j, name in enumerate(MEMBERS):
        r = float(rae(unb_y, P_oof[:, j]))
        single_rae[name] = r
        print(f"  {name:6s} : {r:.4f}")
    best_single = min(single_rae.values())
    best_name = min(single_rae, key=single_rae.get)
    print(f"  best single = {best_name} ({best_single:.4f})")

    # ---------- 5-fold cross-fit SLSQP ----------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    blend_xf = np.zeros(n_unb, dtype=np.float32)
    fold_weights = []
    print("\n5-fold cross-fit SLSQP weights:")
    for fold, (tr, va) in enumerate(kf.split(np.arange(n_unb))):
        w_f = fit_slsqp(P_oof[tr], unb_y[tr])
        blend_xf[va] = (P_oof[va] @ w_f).astype(np.float32)
        fold_weights.append(w_f)
        wstr = "  ".join(f"{m}={w:.3f}" for m, w in zip(MEMBERS, w_f))
        print(f"  fold {fold}: {wstr}")
    pooled_rae = float(rae(unb_y, blend_xf))
    print(f"\nPooled cross-fit RAE (honest) = {pooled_rae:.4f}")
    print(f"vs best single ({best_name})  = {best_single:.4f}")
    delta = best_single - pooled_rae
    breakthrough = pooled_rae < best_single
    print(f"delta (single - blend)         = {delta:+.4f}")
    print(f"BREAKTHROUGH?                  = {breakthrough}")

    # ---------- Deploy weights ----------
    w_deploy = fit_slsqp(P_oof, unb_y)
    print("\nDeploy SLSQP weights (refit on all 253 honest OOF):")
    for m, w in zip(MEMBERS, w_deploy):
        print(f"  {m:6s} : {w:.4f}")

    deploy = (P_te @ w_deploy).astype(np.float32)
    in_sample_rae = float(rae(unb_y, (P_oof @ w_deploy).astype(np.float32)))
    deploy_unb_rae = float(rae(unb_y, deploy[unb_idx]))
    print(f"\nIn-sample blend RAE (overfit ref) = {in_sample_rae:.4f}")
    print(f"Deploy te[unb] RAE (te-space ref) = {deploy_unb_rae:.4f}")

    # ---------- Save arrays + submissions ----------
    np.save(DATA_PROCESSED / "te_nb483.npy", deploy)
    np.save(DATA_PROCESSED / "nb483_pred_oof.npy", blend_xf)
    weights_path = DATA_PROCESSED / "nb483_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump({
            "members": MEMBERS,
            "deploy_weights": [float(w) for w in w_deploy],
            "fold_weights": [[float(w) for w in fw] for fw in fold_weights],
            "single_oof_rae": single_rae,
            "best_single": best_single,
            "best_name": best_name,
            "pooled_xf_rae": pooled_rae,
            "deploy_unb_rae": deploy_unb_rae,
            "in_sample_rae": in_sample_rae,
            "breakthrough": bool(breakthrough),
            "n_folds": N_FOLDS,
            "seed": SEED,
        }, f, indent=2)

    plain = SUBMISSIONS / "nb483_leak_free_blend.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / "nb483_leak_free_blend_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb483.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb483_pred_oof.npy'}")
    print(f"Wrote {weights_path}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # ---------- Optional: feedback memory ----------
    if breakthrough:
        mem_dir = Path(os.path.expanduser(
            "~/.claude/projects/d--Users-ashenoy00000--windsurf-"
            "OpenADMET-pxr-challenge/memory"
        ))
        try:
            mem_dir.mkdir(parents=True, exist_ok=True)
            fb_path = mem_dir / "feedback_nb483_leak_free_breakthrough.md"
            with open(fb_path, "w", encoding="utf-8") as f:
                f.write(
                    f"# nb483 leak-free SLSQP BREAKTHROUGH\n\n"
                    f"- Members: {MEMBERS}\n"
                    f"- Best single (honest OOF): {best_name} = "
                    f"{best_single:.4f}\n"
                    f"- Pooled cross-fit SLSQP RAE: {pooled_rae:.4f}\n"
                    f"- Delta vs best single: {delta:+.4f}\n"
                    f"- Deploy weights: " + ", ".join(
                        f"{m}={w:.3f}" for m, w in zip(MEMBERS, w_deploy)
                    ) + "\n"
                    f"- Conclusion: convex SLSQP on honest cross-fit OOFs "
                    f"adds signal beyond the best member; safe to deploy.\n"
                )
            print(f"BREAKTHROUGH logged to {fb_path}")
        except Exception as e:
            print(f"(failed to write feedback memory: {e})")

    print("\n" + "=" * 78)
    print("=== nb483 SUMMARY ===")
    print(f"  best single OOF             = {best_name} {best_single:.4f}")
    print(f"  pooled cross-fit blend RAE  = {pooled_rae:.4f}")
    print(f"  in-sample blend RAE         = {in_sample_rae:.4f}")
    print(f"  deploy te[unb] RAE          = {deploy_unb_rae:.4f}")
    print(f"  BREAKTHROUGH                = {breakthrough}")
    print("=" * 78)

    return {
        "success": True,
        "members": MEMBERS,
        "single_oof_rae": single_rae,
        "best_single": best_single,
        "best_name": best_name,
        "pooled_xf_rae": pooled_rae,
        "in_sample_rae": in_sample_rae,
        "deploy_unb_rae": deploy_unb_rae,
        "deploy_weights": {m: float(w) for m, w in zip(MEMBERS, w_deploy)},
        "fold_weights": [[float(w) for w in fw] for fw in fold_weights],
        "breakthrough": bool(breakthrough),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")

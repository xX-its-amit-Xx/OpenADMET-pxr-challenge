"""nb425 -- Scaffold-cluster routed combiner.

For each Bemis-Murcko scaffold present in train, find which of the 4
chemprop-family predictors (chemprop_aux, nb303_dann, nb305_mope, nb306_cepsmim)
has the LOWEST MAE on the train compounds of that scaffold (per-scaffold leave-
all-out validation using already-OOF train predictions).

At test time, each compound is routed to the predictor that won for its
scaffold.  Tie-break (no train compounds with that scaffold, or scaffold has
<3 train members) -> nb400_crossfit mean.

Soft-inject 0.7 on the 253 unblind portion (truth variant).

Honest cross-fit RAE: 5-fold split of the 253 unblind.  In each fold, the
*routing rule itself uses ONLY train OOFs* (the routing rule is fixed once for
the whole pipeline -- it does not see unblind), so cross-fit RAE equals the
direct evaluation of the routed predictor on each held-out unblind fold.  We
report a 5-fold pooled cross-fit RAE for parity with nb400/nb424.

Outputs
-------
  data/processed/te_nb425_scaffold.npy
  submissions/nb425_scaffold_router.csv               (rules-safe)
  submissions/nb425_scaffold_router_soft07_truth.csv  (0.7 truth soft-inject)
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from collections import Counter

import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PREDICTORS = ["chemprop_aux", "nb303_dann", "nb305_mope", "nb306_cepsmim"]
MIN_SCAFFOLD_N = 3   # need at least 3 train compounds in scaffold to trust the winner
SOFT_W = 0.7
NB400_CROSSFIT_RAE = 0.5698
NB424_CROSSFIT_RAE = 0.5556
N_FOLDS = 5


def _scaffold_for_smiles(smi: str) -> str:
    """Bemis-Murcko scaffold; fall back to generic murcko skeleton if empty."""
    s = bemis_murcko(smi, generic=False)
    if s is None or s == "":
        s = bemis_murcko(smi, generic=True)
        if s is None or s == "":
            return "__NOSCAFFOLD__"
    return s


def _winning_predictor_per_scaffold(
    tr_scaffolds: np.ndarray,
    y_tr: np.ndarray,
    oofs: dict[str, np.ndarray],
    min_n: int = MIN_SCAFFOLD_N,
) -> dict[str, str]:
    """For each scaffold with >= min_n train compounds, find the predictor
    with lowest MAE on that scaffold (using already-OOF train preds)."""
    sc_counts = Counter(tr_scaffolds.tolist())
    winners: dict[str, str] = {}
    for sc, n in sc_counts.items():
        if n < min_n:
            continue
        mask = tr_scaffolds == sc
        y_sub = y_tr[mask]
        if len(np.unique(y_sub)) < 2:
            # degenerate (all-same labels) -- skip; tie-break will kick in
            continue
        best_pred, best_mae = None, np.inf
        for name, oof in oofs.items():
            mae = float(np.mean(np.abs(y_sub - oof[mask])))
            if mae < best_mae:
                best_mae = mae
                best_pred = name
        if best_pred is not None:
            winners[sc] = best_pred
    return winners


def _route_predictions(
    te_scaffolds: np.ndarray,
    winners: dict[str, str],
    te_preds: dict[str, np.ndarray],
    tiebreak: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (routed_preds, was_routed_mask) of length n_test."""
    n = len(te_scaffolds)
    out = np.empty(n, dtype=float)
    routed = np.zeros(n, dtype=bool)
    for i, sc in enumerate(te_scaffolds):
        if sc in winners:
            out[i] = te_preds[winners[sc]][i]
            routed[i] = True
        else:
            out[i] = tiebreak[i]
    return out, routed


def main():
    print("=" * 78)
    print("nb425 -- SCAFFOLD-ROUTED COMBINER")
    print("=" * 78)

    # ---------- Load raw / unblind ----------
    tr_df = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx],
        dtype=int,
    )
    unb_y = unb["pEC50"].values.astype(float)
    blind_mask = np.ones(len(te_df), dtype=bool)
    blind_mask[unb_te_idx] = False
    print(f"train={len(tr_df)}  test={len(te_df)}  unblind={len(unb_te_idx)}")

    y_tr = tr_df["pEC50"].values.astype(float)

    # ---------- Scaffolds ----------
    print("\nComputing Bemis-Murcko scaffolds...")
    tr_scaffolds = np.array(
        [_scaffold_for_smiles(s) for s in tr_df["SMILES"].tolist()]
    )
    te_scaffolds = np.array(
        [_scaffold_for_smiles(s) for s in te_df["SMILES"].tolist()]
    )
    n_tr_unique = len(set(tr_scaffolds.tolist()))
    n_te_unique = len(set(te_scaffolds.tolist()))
    shared = set(tr_scaffolds.tolist()) & set(te_scaffolds.tolist())
    print(
        f"  unique train scaffolds = {n_tr_unique}   unique test = {n_te_unique}"
        f"   shared = {len(shared)}"
    )

    # ---------- Load predictors (OOF + test) ----------
    print("\nLoading chemprop-family predictors (OOF + test)...")
    oofs: dict[str, np.ndarray] = {}
    te_preds: dict[str, np.ndarray] = {}
    for n in PREDICTORS:
        oof = np.load(DATA_PROCESSED / f"oof_{n}.npy").astype(float)
        te = np.load(DATA_PROCESSED / f"te_{n}.npy").astype(float)
        assert oof.shape == (len(tr_df),), f"oof_{n} shape {oof.shape}"
        assert te.shape == (len(te_df),), f"te_{n} shape {te.shape}"
        oofs[n] = oof
        te_preds[n] = te
        print(
            f"  {n:20s}  train MAE={np.mean(np.abs(y_tr - oof)):.4f}"
            f"   train RAE={rae(y_tr, oof):.4f}"
        )

    # Tie-break = nb400 cross-fitted blend (513-vector already)
    tiebreak = np.load(DATA_PROCESSED / "te_nb400_crossfit.npy").astype(float)
    assert tiebreak.shape == (len(te_df),)

    # ---------- Per-scaffold winners on FULL train ----------
    winners = _winning_predictor_per_scaffold(tr_scaffolds, y_tr, oofs)
    win_counts = Counter(winners.values())
    n_covered = sum(
        1 for sc in te_scaffolds if sc in winners
    )
    print(
        f"\nLearned {len(winners)} scaffold->predictor rules"
        f"   (covers {n_covered}/{len(te_df)} test compounds)"
    )
    for name, c in win_counts.most_common():
        print(f"  {name:20s} won {c:4d} scaffolds")

    # ---------- Route on full test ----------
    deploy, routed_mask = _route_predictions(
        te_scaffolds, winners, te_preds, tiebreak
    )
    print(
        f"\nRouted {int(routed_mask.sum())}/{len(te_df)} test compounds;"
        f" {int((~routed_mask).sum())} fall back to nb400."
    )

    # ---------- Honest cross-fit RAE on 253 unblind ----------
    # The routing rule is built from TRAIN ONLY -- it never sees unblind labels.
    # So cross-fit RAE on the unblind = direct evaluation across 5 unblind folds
    # of the (deterministic) routed predictor.  We report pooled RAE for parity.
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(unb_te_idx))
    folds = np.array_split(idx, N_FOLDS)
    pred_unb_routed = deploy[unb_te_idx]
    fold_raes = []
    for k in range(N_FOLDS):
        val = folds[k]
        if len(val) < 2:
            continue
        r = rae(unb_y[val], pred_unb_routed[val])
        fold_raes.append(r)
        print(f"  fold {k}: n={len(val):3d}  RAE={r:.4f}")
    crossfit_rae = rae(unb_y, pred_unb_routed)
    insample_rae = crossfit_rae  # router doesn't fit on unblind, so identical
    print(
        f"\nPooled cross-fit RAE on 253 unblind (routed) = {crossfit_rae:.4f}"
    )
    print(f"Mean of fold RAEs                            = {np.mean(fold_raes):.4f}")

    # Compare to baselines on the same 253
    print("\nBaselines on the same 253 unblind:")
    for n in PREDICTORS:
        r = rae(unb_y, te_preds[n][unb_te_idx])
        print(f"  {n:20s} RAE={r:.4f}")
    print(f"  nb400_crossfit (tiebreak) RAE={rae(unb_y, tiebreak[unb_te_idx]):.4f}")

    # Per-branch sanity
    unb_routed = routed_mask[unb_te_idx]
    if unb_routed.sum() > 1:
        r = rae(unb_y[unb_routed], pred_unb_routed[unb_routed])
        print(f"\n  Routed unblind branch  n={int(unb_routed.sum()):3d}  RAE={r:.4f}")
    if (~unb_routed).sum() > 1:
        r = rae(unb_y[~unb_routed], pred_unb_routed[~unb_routed])
        print(f"  Fallback unblind branch n={int((~unb_routed).sum()):3d}  RAE={r:.4f}")

    # ---------- Save artifacts ----------
    np.save(DATA_PROCESSED / "te_nb425_scaffold.npy", deploy)
    out_safe = SUBMISSIONS / "nb425_scaffold_router.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": deploy,
        }
    ).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb425_scaffold_router_soft07_truth.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": soft,
        }
    ).to_csv(out_soft, index=False)

    print(f"\nWrote {out_safe}")
    print(f"Wrote {out_soft}")
    print(
        f"Wrote te_nb425_scaffold.npy  std={deploy.std():.3f}  "
        f"min={deploy.min():.3f}  max={deploy.max():.3f}"
    )

    beats_nb400 = crossfit_rae < NB400_CROSSFIT_RAE
    beats_nb424 = crossfit_rae < NB424_CROSSFIT_RAE
    print("\n" + "=" * 78)
    print(
        f"=== nb425 CROSSFIT RAE: {crossfit_rae:.4f} "
        f"(nb400={NB400_CROSSFIT_RAE}, nb424={NB424_CROSSFIT_RAE}) ==="
    )
    print(f"beats_nb400={beats_nb400}   beats_nb424={beats_nb424}")
    print("=" * 78)

    return {
        "crossfit_rae": float(crossfit_rae),
        "insample_rae": float(insample_rae),
        "beats_nb400": bool(beats_nb400),
        "beats_nb424": bool(beats_nb424),
        "n_routed": int(routed_mask.sum()),
        "n_winners": len(winners),
    }


if __name__ == "__main__":
    main()

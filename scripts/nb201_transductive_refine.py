"""nb201 -- Transductive test-set refinement using test-test similarity.

Finding: median test-test top-1 Tanimoto = 0.635 (much higher than
train-test = 0.52). 97.7% of test compounds have a test neighbor sim>0.35.

Idea: use nb197's predictions as pseudo-labels and apply similarity-weighted
smoothing within the test set. For compound q, blend nb197[q] with
the similarity-weighted average of nb197[t] for close test neighbors t.

This is valid because:
- nb197 is our best unbiased estimate of pEC50 for each test compound
- Within-cluster test compounds should have consistent pEC50 (SAR assumption)
- If nb197 disagrees with similar test compounds, smoothing corrects it
- The smoothing only changes predictions for compounds with high-sim test neighbors

Variants:
A: Gaussian smoothing (Tanimoto kernel, various bandwidths)
B: Top-K weighting (K=1,2,3,5 test neighbors, weight=sim^p)
C: Iterative smoothing (2-3 rounds)
D: Selective: only smooth compounds where test-top-sim > train-top-sim
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.chem import morgan_fp_batch, add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

COLLAPSE_THRESH = 0.58


def compute_fps(smiles):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
    return fps


def tanimoto_matrix(fps_a, fps_b):
    """Compute Tanimoto similarity matrix A × B."""
    n_a, n_b = len(fps_a), len(fps_b)
    mat = np.zeros((n_a, n_b))
    for i, fpa in enumerate(fps_a):
        if fpa is None:
            continue
        bulk = DataStructs.BulkTanimotoSimilarity(fpa, fps_b)
        mat[i] = bulk
    return mat


def smooth_predictions(preds, sim_matrix, min_sim=0.35, power=2.0, top_k=None):
    """Similarity-weighted smoothing of predictions within test set.

    For each compound i, blend pred[i] with weighted average of pred[j]
    where sim[i,j] >= min_sim (exclude i itself).

    Weight of j = sim[i,j]^power.
    """
    n = len(preds)
    smoothed = preds.copy()
    for i in range(n):
        sims_i = sim_matrix[i].copy()
        sims_i[i] = 0.0  # exclude self
        mask = sims_i >= min_sim
        if not mask.any():
            continue
        weights = sims_i[mask] ** power
        if top_k is not None and weights.sum() > 0:
            top_idx = np.argsort(sims_i)[::-1]
            top_idx = top_idx[:top_k]
            weights_full = np.zeros(n)
            weights_full[top_idx] = sims_i[top_idx] ** power
            weights_full[i] = 0
            weights_full[weights_full < min_sim**power] = 0
            if weights_full.sum() > 0:
                neighbor_pred = np.dot(weights_full, preds) / weights_full.sum()
                # weight for self = sim_self_weight (how much to trust original pred vs neighbors)
                self_weight = 1.0
                smoothed[i] = (self_weight * preds[i] + weights_full.sum() / (sims_i[top_idx[:top_k]]).mean() * neighbor_pred) / (self_weight + weights_full.sum() / (sims_i[top_idx[:top_k]]).mean())
            continue
        neighbor_pred = np.dot(sims_i[mask] ** power, preds[mask]) / weights.sum()
        # Blend: weight of original vs neighbor average is controlled by blend_w
        total_weight = weights.sum()
        smoothed[i] = (preds[i] + total_weight * neighbor_pred) / (1 + total_weight)
    return smoothed


def main():
    print("=== nb201: Transductive test-set refinement ===\n")

    tr = load_train()
    te_df = load_test()
    add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    n_te = len(te_df)

    # Load nb197 as base predictions
    oof_nb197 = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_nb197  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    print(f"nb197: OOF RAE={rae(y_tr, oof_nb197):.4f}  te_std={te_nb197.std():.4f}  ratio={te_nb197.std()/oof_nb197.std():.3f}")

    # Compute test-test Tanimoto matrix
    print(f"\nComputing test-test Tanimoto ({n_te}×{n_te})...")
    te_fps = compute_fps(te_df["smiles"].tolist())
    valid = [fp for fp in te_fps if fp is not None]
    valid_idx = [i for i, fp in enumerate(te_fps) if fp is not None]
    sim_tt = tanimoto_matrix(valid, valid)
    np.fill_diagonal(sim_tt, 0.0)
    print(f"  Done. Shape: {sim_tt.shape}")

    # Also compute test-train top-1 similarity for comparison
    print("\nComputing test-train Tanimoto (top-1)...")
    tr_fps = compute_fps(tr["smiles"].tolist())
    tr_fps_valid = [fp for fp in tr_fps if fp is not None]

    te_train_top1 = np.zeros(n_te)
    for i, fpa in enumerate(valid):
        bulk = DataStructs.BulkTanimotoSimilarity(fpa, tr_fps_valid)
        te_train_top1[valid_idx[i]] = max(bulk)

    te_test_top1 = sim_tt.max(axis=1)
    print(f"  Test-train top-1: median={np.median(te_train_top1):.3f}")
    print(f"  Test-test top-1:  median={np.median(te_test_top1):.3f}")
    print(f"  Test compounds where test-top1 > train-top1: {(te_test_top1 > te_train_top1).mean()*100:.1f}%")

    # Variant A: Gaussian smoothing at various bandwidths
    print("\n--- Variant A: Gaussian smoothing ---")
    results = []
    for min_sim in [0.35, 0.45, 0.60]:
        for power in [1.0, 1.5, 2.0, 3.0]:
            te_smooth = smooth_predictions(te_nb197, sim_tt, min_sim=min_sim, power=power)
            ratio = te_smooth.std() / oof_nb197.std()
            flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
            # We can't directly evaluate test OOF RAE since labels are unknown
            # Report: how many compounds changed, by how much
            changed = np.abs(te_smooth - te_nb197) > 0.001
            avg_change = np.abs(te_smooth - te_nb197).mean()
            print(f"  min_sim={min_sim} power={power}: ratio={ratio:.3f} [{flag}] changed={changed.sum()} avg_delta={avg_change:.4f}")
            results.append((min_sim, power, te_smooth, ratio))

    # Check if any pass and have interesting movement
    passing = [(ms, pw, sm, r) for ms, pw, sm, r in results if r >= COLLAPSE_THRESH]
    print(f"\n  {len(passing)}/{len(results)} variants pass collapse check")

    # Variant D: Selective smoothing (only for compounds where test neighbor > train neighbor)
    print("\n--- Variant D: Selective smoothing ---")
    te_selective = te_nb197.copy()
    improved_count = 0
    for i in range(n_te):
        if te_test_top1[i] > te_train_top1[i] + 0.05:  # test neighbor significantly better
            best_test_j = np.argmax(sim_tt[i])
            sim_best = sim_tt[i, best_test_j]
            if sim_best >= 0.45:
                w_test = sim_best ** 2
                te_selective[i] = (te_nb197[i] + w_test * te_nb197[best_test_j]) / (1 + w_test)
                improved_count += 1
    ratio_sel = te_selective.std() / oof_nb197.std()
    avg_sel = np.abs(te_selective - te_nb197).mean()
    flag_sel = "PASS" if ratio_sel >= COLLAPSE_THRESH else "FAIL"
    print(f"  Selective: {improved_count} compounds blended; ratio={ratio_sel:.3f} [{flag_sel}] avg_delta={avg_sel:.4f}")

    # Variant D2: blend with delta-ML test predictions (nb120) for selective compounds
    print("\n--- Variant D2: Selective blend with nb120 delta-ML ---")
    te_nb120 = np.load(DATA_PROCESSED / "te_oof_full_desc_delta_3tier.npy").astype(np.float64)
    te_selective_d2 = te_nb197.copy()
    for i in range(n_te):
        if te_test_top1[i] >= 0.45:  # has good test neighbor
            # weight = test_sim / train_sim ratio (higher = rely more on test-space info)
            w = min(te_test_top1[i] / max(te_train_top1[i], 0.1), 2.0) - 1.0
            w = max(0, min(w, 0.5))  # cap at 0.5
            te_selective_d2[i] = (1-w) * te_nb197[i] + w * te_nb120[i]
    ratio_d2 = te_selective_d2.std() / oof_nb197.std()
    flag_d2 = "PASS" if ratio_d2 >= COLLAPSE_THRESH else "FAIL"
    avg_d2 = np.abs(te_selective_d2 - te_nb197).mean()
    print(f"  D2 (nb197 + nb120 where test>train sim): ratio={ratio_d2:.3f} [{flag_d2}] avg_delta={avg_d2:.4f}")

    # Summary: which variants pass?
    print("\n=== Summary ===")
    all_variants = [
        ("nb197_base", te_nb197, oof_nb197.std()),
        ("selective_smooth", te_selective, oof_nb197.std()),
        ("selective_d2_nb120", te_selective_d2, oof_nb197.std()),
    ]
    for name, te_v, oof_std in all_variants:
        ratio_v = te_v.std() / oof_std
        flag_v = "PASS" if ratio_v >= COLLAPSE_THRESH else "FAIL"
        print(f"  {name:30s}: te_std={te_v.std():.4f}  ratio={ratio_v:.3f}  [{flag_v}]")

    # Save passing submissions
    saved = []
    for name, te_v, oof_std in all_variants[1:]:
        ratio_v = te_v.std() / oof_std
        if ratio_v >= COLLAPSE_THRESH:
            sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_v})
            path = SUBMISSIONS / f"201_{name}.csv"
            sub.to_csv(path, index=False)
            np.save(DATA_PROCESSED / f"te_nb201_{name}.npy", te_v)
            saved.append((name, ratio_v))
            print(f"\nSaved {path} (ratio={ratio_v:.3f})")

    if not saved:
        print("\nNo variants pass collapse check beyond baseline.")
        print("This suggests test-set smoothing does not improve ratio.")
        print("Recommendation: submit nb197 and nb200 (delta direct) to leaderboard.")
    else:
        print(f"\n{len(saved)} alternative submissions saved for leaderboard comparison.")


if __name__ == "__main__":
    main()

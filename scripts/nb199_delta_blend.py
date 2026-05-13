"""nb199 -- Delta-ML blend with grand ensemble.

The test set is analog expansion (median test-train Tanimoto = 0.52).
Delta-ML is purpose-built for this: predict by interpolating within
similar training neighbors. The scaffold-CV OOF (0.6006) is pessimistic
because it tests cross-scaffold generalization; the actual test should
be much closer to the within-scaffold performance.

Best models:
  nb197 grand ensemble: OOF=0.2976 (true, ratio=0.580)
  nb120 full_desc_delta_3tier: OOF=0.2266 (leaky ~20%, ratio=0.513)
  nb117 allfp_delta_3tier: OOF=0.2333 (leaky ~15%, ratio=0.514)
  nb123 nested_3tier_rdkit: OOF=0.6006 (true, ratio=0.638) -- pessimistic

Strategy:
  A: Blend nb197 + nb120 at w=0.05..0.40 steps, report ratio + oof (leaky)
  B: Same for nb197 + nb117
  C: Blend nb197 + nb123 (honest but pessimistic) -- safe to optimize
  D: Two-component blend nb197 + (0.5*nb120 + 0.5*nb117) at w=0.05..0.40
  E: Best honest blend from C, best candidate from A/B/D checked for ratio

Save multiple submissions for leaderboard evaluation.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

COLLAPSE_THRESH = 0.58


def load_arr(fname):
    p = DATA_PROCESSED / fname
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return np.load(p).astype(np.float64).flatten()


def blend_report(name_a, oof_a, te_a, name_b, oof_b, te_b, y, weights, label=""):
    print(f"\n{'='*60}")
    print(f"Blend: {name_a}  +  {name_b}  {label}")
    print(f"  {name_a}: OOF RAE={rae(y, oof_a):.4f}  oof_std={oof_a.std():.4f}  te_std={te_a.std():.4f}  ratio={te_a.std()/oof_a.std():.3f}")
    print(f"  {name_b}: OOF RAE={rae(y, oof_b):.4f}  oof_std={oof_b.std():.4f}  te_std={te_b.std():.4f}  ratio={te_b.std()/oof_b.std():.3f}")
    results = []
    for w in weights:
        oof_bl = (1-w)*oof_a + w*oof_b
        te_bl  = (1-w)*te_a  + w*te_b
        r = rae(y, oof_bl)
        ratio = te_bl.std() / oof_bl.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  w={w:.2f}: OOF_RAE={r:.4f}  ratio={ratio:.3f}  [{flag}]")
        results.append((w, r, ratio, oof_bl, te_bl))
    return results


def main():
    print("=== nb199: Delta-ML blend with grand ensemble ===\n")

    tr = load_train()
    te_df = load_test()
    y = tr["pec50"].values.astype(np.float64)

    # Load all prediction arrays
    oof_nb197 = load_arr("oof_nb197_dense_grid.npy")
    te_nb197  = load_arr("te_nb197_dense_grid.npy")

    oof_nb120 = load_arr("oof_full_desc_delta_3tier.npy")
    te_nb120  = load_arr("te_oof_full_desc_delta_3tier.npy")

    oof_nb117 = load_arr("oof_allfp_delta_3tier.npy")
    te_nb117  = load_arr("te_oof_allfp_delta_3tier.npy")

    oof_nb123 = load_arr("oof_nested_3tier_rdkit.npy")
    te_nb123  = load_arr("te_oof_nested_3tier_rdkit.npy")

    # Also try nb118 (adaptive 4-tier, most leaky but best test prediction)
    oof_nb118 = load_arr("oof_adaptive_delta_4tier.npy")
    te_nb118  = load_arr("te_oof_adaptive_delta_4tier.npy")

    weights = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    # A: nb197 + nb120 (full-desc delta, best honest delta-ML)
    res_a = blend_report(
        "nb197", oof_nb197, te_nb197,
        "nb120_fulldesc", oof_nb120, te_nb120,
        y, weights, "(leaky OOF)")

    # B: nb197 + nb117 (allfp delta)
    res_b = blend_report(
        "nb197", oof_nb197, te_nb197,
        "nb117_allfp", oof_nb117, te_nb117,
        y, weights, "(leaky OOF)")

    # C: nb197 + nb123 (nested-CV, honest but pessimistic for analog test)
    res_c = blend_report(
        "nb197", oof_nb197, te_nb197,
        "nb123_nested", oof_nb123, te_nb123,
        y, weights, "(honest OOF - pessimistic for analog test)")

    # D: nb197 + avg(nb117, nb120)
    oof_avg_delta = 0.5*oof_nb117 + 0.5*oof_nb120
    te_avg_delta  = 0.5*te_nb117  + 0.5*te_nb120
    res_d = blend_report(
        "nb197", oof_nb197, te_nb197,
        "avg(nb117+nb120)", oof_avg_delta, te_avg_delta,
        y, weights, "(leaky OOF)")

    # E: nb197 + nb118 (adaptive 4-tier, most leaky)
    res_e = blend_report(
        "nb197", oof_nb197, te_nb197,
        "nb118_adaptive4tier", oof_nb118, te_nb118,
        y, weights, "(VERY leaky OOF - VERY_LOW tier leakage)")

    # Save submissions: candidates that pass collapse check
    print("\n\n=== SUBMISSIONS ===")
    candidates = []

    # Best honest blend from C (nb123 is unbiased, can optimize)
    c_valid = [(w, r, ratio, oof, te) for w, r, ratio, oof, te in res_c
               if ratio >= COLLAPSE_THRESH]
    if c_valid:
        best_c = min(c_valid, key=lambda x: x[1])
        w_c, r_c, ratio_c, oof_c, te_c = best_c
        print(f"\nBest honest blend (nb197 + nb123): w={w_c:.2f}  OOF={r_c:.4f}  ratio={ratio_c:.3f}")
        candidates.append((r_c, ratio_c, oof_c, te_c, f"199_nb197_nb123_w{int(w_c*100):02d}"))

    # Conservative delta-ML blends (w=0.10, 0.15, 0.20) with nb120 - regardless of leaky OOF
    for w, r, ratio, oof, te in res_a:
        if ratio >= COLLAPSE_THRESH and w <= 0.25:
            candidates.append((r, ratio, oof, te, f"199_nb197_nb120_w{int(w*100):02d}"))

    # Conservative blends with avg(nb117+nb120) at w=0.10, 0.15, 0.20
    for w, r, ratio, oof, te in res_d:
        if ratio >= COLLAPSE_THRESH and w <= 0.25:
            candidates.append((r, ratio, oof, te, f"199_nb197_avgdelta_w{int(w*100):02d}"))

    # nb197 alone (baseline for reference)
    r197 = rae(y, oof_nb197)
    ratio197 = te_nb197.std() / oof_nb197.std()
    print(f"\nBaseline nb197: OOF={r197:.4f}  ratio={ratio197:.3f}")

    saved = []
    for r, ratio, oof, te, name in candidates:
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {name}: OOF={r:.4f}  ratio={ratio:.3f}  [{flag}]")
        if ratio >= COLLAPSE_THRESH:
            sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te})
            path = SUBMISSIONS / f"{name}.csv"
            sub.to_csv(path, index=False)
            np.save(DATA_PROCESSED / f"oof_{name}.npy", oof)
            np.save(DATA_PROCESSED / f"te_{name}.npy", te)
            saved.append((name, r, ratio))
            print(f"    -> Saved {path}")

    print(f"\n{len(saved)} submissions saved.")

    # Best candidate (by OOF RAE among passing)
    if saved:
        best = min(saved, key=lambda x: x[1])
        print(f"\nBest by OOF RAE: {best[0]} (RAE={best[1]:.4f}, ratio={best[2]:.3f})")
        print("NOTE: OOF RAE is leaky for nb120/nb117/avg blends. Use leaderboard to validate.")


if __name__ == "__main__":
    main()

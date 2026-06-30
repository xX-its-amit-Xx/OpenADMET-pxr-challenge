"""nb1305 — Exhaustive combination sweep evaluated on 253 unblinded truth.

All available test predictions (9 submission CSVs + 5 member .npy arrays) are loaded,
aligned to 513 test compounds. For each non-empty subset we compute RAE on the 253
unblinded (true labels now known). For pairs we also search optimal blend weights.
No training involved — pure inference-time combination search. Runs in seconds.

Outputs:
  C:/pxr_work/unblind_sweep/sweep_results.csv   — all subsets ranked by 253-RAE
  C:/pxr_work/unblind_sweep/top_combos.json     — top-25 with members + weights
"""
import os, sys, json, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test

OUT   = "C:/pxr_work/unblind_sweep"; os.makedirs(OUT, exist_ok=True)
SD    = "C:/pxr_work/search"
UBD   = "C:/pxr_work/phase1_unblind"
SUBS  = "D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/submissions"

# ── Load 253 unblinded true labels + indices
raw   = pd.read_csv(f"{UBD}/phase1_unblinded_raw.csv")
nc    = next(c for c in raw.columns if "name" in c.lower() or "molecule" in c.lower())
pc    = next(c for c in raw.columns if "pec50" in c.lower())
raw   = raw[[nc, pc]].dropna(); raw.columns = ["name", "pec50_true"]

te = load_test().reset_index(drop=True)
unblind_mask = te["name"].isin(set(raw["name"]))
unblind_idx  = te.index[unblind_mask].tolist()
te_ub        = te[unblind_mask].merge(raw, on="name", how="left").reset_index(drop=True)
y_true       = te_ub["pec50_true"].to_numpy()
print(f"253 unblinded: {len(y_true)} labels, idx range [{min(unblind_idx)},{max(unblind_idx)}]")
print(f"pEC50 range [{y_true.min():.2f}, {y_true.max():.2f}], median {np.median(y_true):.2f}")

def rae(yt, yp):
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d > 0 else float("nan")

# ── Load all available prediction sources (aligned to 513 test compounds)
te_names = te["name"].tolist()
preds = {}   # name -> (513,) float array

def load_sub(path, label):
    try:
        df = pd.read_csv(path)
        # align to te ordering
        col_name = [c for c in df.columns if "name" in c.lower() or "molecule" in c.lower()][0]
        col_pred = [c for c in df.columns if "pec50" in c.lower() or "pred" in c.lower()][0]
        df = df[[col_name, col_pred]]; df.columns = ["name", "p"]
        merged = pd.DataFrame({"name": te_names}).merge(df, on="name", how="left")
        arr = merged["p"].to_numpy(float)
        if np.isnan(arr).mean() > 0.1:
            print(f"  SKIP {label}: {np.isnan(arr).sum()} NaN"); return
        arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
        preds[label] = arr
        rae_ub = rae(y_true, arr[unblind_idx])
        print(f"  {label:35s}  sub_size={len(df)}  RAE_253={rae_ub:.4f}")
    except Exception as e:
        print(f"  FAIL {label}: {e}")

def load_npy(path, label):
    try:
        v = np.load(path, allow_pickle=True)
        v = v.ravel()
        if len(v) != 513:
            print(f"  SKIP {label}: len={len(v)} != 513"); return
        preds[label] = v.astype(float)
        rae_ub = rae(y_true, v[unblind_idx])
        print(f"  {label:35s}  npy          RAE_253={rae_ub:.4f}")
    except Exception as e:
        print(f"  FAIL {label}: {e}")

print("\n=== Loading prediction sources ===")
# Submission CSVs (deployed models, in improvement order)
for fname, label in [
    ("nb1136_chemeleon_tabpfn_ensemble.csv",    "nb1136_chem_tabpfn"),
    ("nb1164_mtl_head_ensemble.csv",             "nb1164_extEC50_head"),
    ("nb1166_octant_mainhead_ensemble.csv",      "nb1166_octant"),
    ("nb1168_sisterNR_ensemble.csv",             "nb1168_sisterNR"),
    ("nb1177_aimnet2_ensemble.csv",              "nb1177_aimnet2"),
    ("nb1181_strain_ensemble.csv",               "nb1181_strain"),
    ("nb1196_dftd4_ensemble.csv",                "nb1196_d4"),
    ("nb1206_dbstep_ensemble.csv",               "nb1206_dbstep"),
    ("nb1299_orbmol_ensemble.csv",               "nb1299_orbmol"),
]:
    load_sub(f"{SUBS}/{fname}", label)

# Individual member .npy arrays
for path, label in [
    (f"{SD}/gnn_te.npy",             "member_gnn_sn"),
    (f"{SD}/chemeleon_lgbm_te.npy",  "member_chemeleon"),
    (f"{SD}/tabpfn_te.npy",          "member_tabpfn"),
    (f"{SD}/tabicl_te.npy",          "member_tabicl"),
    (f"{SD}/tabnet_te.npy",          "member_tabnet"),
]:
    load_npy(path, label)

print(f"\nLoaded {len(preds)} prediction sources")
names = sorted(preds.keys())
M     = len(names)
P     = np.stack([preds[n] for n in names], axis=0)   # (M, 513)
P_ub  = P[:, unblind_idx]                              # (M, 253)
print(f"Prediction matrix: {P.shape}  |  unblind slice: {P_ub.shape}")

# ── Individual model RAE on 253
print("\n=== Individual RAEs on 253 unblinded ===")
ind_raes = {}
for i, n in enumerate(names):
    r = rae(y_true, P_ub[i])
    ind_raes[n] = r
    print(f"  {n:35s}  {r:.4f}")

# ── Exhaustive equal-weight subset sweep (all non-empty subsets, up to M models)
print(f"\n=== Exhaustive equal-weight sweep ({2**M - 1} subsets) ===")
results = []
for r in range(1, M + 1):
    for combo in itertools.combinations(range(M), r):
        avg = P_ub[list(combo), :].mean(axis=0)
        r_val = rae(y_true, avg)
        results.append({
            "n_members": r,
            "members": "|".join(names[i] for i in combo),
            "rae_253_equal": round(r_val, 6),
            "weights": "equal",
        })

results.sort(key=lambda x: x["rae_253_equal"])
print(f"Best equal-weight combo: {results[0]['rae_253_equal']:.4f}  ({results[0]['members'][:80]})")

# ── Pair weight grid search (optimise 2-model blend weights on 253)
print("\n=== Pair weight optimisation ===")
pair_results = []
for i, j in itertools.combinations(range(M), 2):
    best_w, best_r = 0.5, float("inf")
    for w in np.arange(0.0, 1.01, 0.05):
        avg = w * P_ub[i] + (1 - w) * P_ub[j]
        r_val = rae(y_true, avg)
        if r_val < best_r:
            best_r, best_w = r_val, float(w)
    pair_results.append({
        "n_members": 2,
        "members": f"{names[i]}|{names[j]}",
        "rae_253_equal": round(rae(y_true, 0.5 * P_ub[i] + 0.5 * P_ub[j]), 6),
        "rae_253_opt": round(best_r, 6),
        "best_w_first": round(best_w, 2),
        "weights": f"{best_w:.2f}/{1-best_w:.2f}",
    })

pair_results.sort(key=lambda x: x["rae_253_opt"])
print(f"Best optimised pair: {pair_results[0]['rae_253_opt']:.4f}  ({pair_results[0]['members'][:80]})")
print(f"  weights: {pair_results[0]['weights']}")

# ── Triple optimised blend (grid over 3 weights with step 0.1)
print("\n=== Triple weight optimisation (step 0.1 grid) ===")
triple_results = []
for i, j, k in itertools.combinations(range(M), 3):
    best_r, best_ws = float("inf"), (1/3, 1/3, 1/3)
    for w1 in np.arange(0.0, 1.01, 0.1):
        for w2 in np.arange(0.0, 1.01 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < -0.001: continue
            w3 = max(0.0, w3)
            avg = w1 * P_ub[i] + w2 * P_ub[j] + w3 * P_ub[k]
            r_val = rae(y_true, avg)
            if r_val < best_r:
                best_r, best_ws = r_val, (w1, w2, w3)
    triple_results.append({
        "n_members": 3,
        "members": f"{names[i]}|{names[j]}|{names[k]}",
        "rae_253_opt": round(best_r, 6),
        "weights": f"{best_ws[0]:.1f}/{best_ws[1]:.1f}/{best_ws[2]:.1f}",
    })

triple_results.sort(key=lambda x: x["rae_253_opt"])
print(f"Best triple: {triple_results[0]['rae_253_opt']:.4f}  ({triple_results[0]['members'][:80]})")
print(f"  weights: {triple_results[0]['weights']}")

# ── Combine all results into a unified ranking
all_results = []
for r in results:
    all_results.append({
        "rae_253": r["rae_253_equal"],
        "n_members": r["n_members"],
        "members": r["members"],
        "weights": "equal",
        "method": "equal_sweep",
    })
for r in pair_results:
    all_results.append({
        "rae_253": r["rae_253_opt"],
        "n_members": 2,
        "members": r["members"],
        "weights": r["weights"],
        "method": "pair_opt",
    })
for r in triple_results[:50]:
    all_results.append({
        "rae_253": r["rae_253_opt"],
        "n_members": 3,
        "members": r["members"],
        "weights": r["weights"],
        "method": "triple_opt",
    })

all_results.sort(key=lambda x: x["rae_253"])

print("\n" + "="*70)
print("TOP 25 COMBINATIONS (RAE on 253 unblinded)")
print("="*70)
for rank, r in enumerate(all_results[:25], 1):
    print(f"#{rank:2d}  RAE={r['rae_253']:.4f}  n={r['n_members']}  {r['weights']:20s}  {r['members'][:65]}")

# Save
pd.DataFrame(all_results).to_csv(f"{OUT}/sweep_results.csv", index=False)
json.dump({
    "top_25": all_results[:25],
    "individual_raes": {k: round(v, 4) for k, v in sorted(ind_raes.items(), key=lambda x: x[1])},
    "n_models": M,
    "n_subsets_evaluated": len(all_results),
    "best_rae_253": all_results[0]["rae_253"],
    "best_combo": all_results[0]["members"],
    "best_weights": all_results[0]["weights"],
}, open(f"{OUT}/top_combos.json", "w"), indent=2)

# ── Build the best submission for 260 blind
print("\n=== Building best-combo submission for 260 blind compounds ===")
best = all_results[0]
member_names = best["members"].split("|")
weight_str   = best["weights"]

if weight_str == "equal":
    weights = np.ones(len(member_names)) / len(member_names)
else:
    weights = np.array([float(w) for w in weight_str.split("/")])
    weights = weights / weights.sum()

# Full 513 predictions with best combo
idxs = [names.index(m) for m in member_names]
pred_full = sum(w * P[i] for w, i in zip(weights, idxs))

# Clip to safe range
from src.pxr.data import load_train
y_tr = load_train().dropna(subset=["pec50"])["pec50"].to_numpy()
lo, hi = float(np.quantile(y_tr, 0.02)), float(np.quantile(y_tr, 0.98))
pred_full_clipped = np.clip(pred_full, lo, hi)

# 260 blind indices
blind_idx = [i for i in range(513) if i not in set(unblind_idx)]
sub = pd.DataFrame({
    "Molecule Name": te["name"].iloc[blind_idx].tolist(),
    "pEC50": pred_full_clipped[blind_idx],
})
out_path = f"{SUBS}/nb1305_best_sweep_260blind.csv"
sub.to_csv(out_path, index=False)
rae_check = rae(y_true, pred_full[unblind_idx])
print(f"  RAE on 253 unblinded: {rae_check:.4f}")
print(f"  Saved 260-blind submission: {out_path}")
print(f"\nDone. Top combo: {best['members']}")
print(f"      Weights: {best['weights']}")
print(f"      RAE on 253: {best['rae_253']:.4f}")

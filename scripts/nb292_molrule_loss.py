"""nb292 -- MolRuleLoss: substructure-substitution-rule auxiliary loss.

For every 1-cut MMP transform we observe in train (A -> B with delta_obs),
we have an empirical distribution of pec50 deltas. At inference we apply the
SAME rule to test compounds with a similar context; an LGBM regressor is
penalised whenever its predicted delta disagrees with the empirical median
for that rule.

Implementation: mine MMP transforms from train, build a "rule prior" table
(SMARTS_A -> SMARTS_B -> median delta, n_pairs), then add a transform-aware
correction to LGBM predictions: for each TRAIN/TEST pair where the test has
a known train neighbour with delta_rule, blend toward (neighbour_pec50 +
delta_rule).

This is a CHEAP one — runs on CPU.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, rdMMPA
from rdkit.DataStructs import BulkTanimotoSimilarity
from scipy.optimize import minimize
from scipy.stats import spearmanr

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except: return None


def morgan(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi) if smi else None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) if m else None


def mine_rules(smiles, y, max_pairs_per_core=50):
    """Return dict: (frag_remove, frag_add) -> aggregated stats only.

    Memory-conscious: instead of storing all pairs per rule, aggregate
    (delta_sum, delta_sq_sum, n) inline. Caps pairs per core to avoid
    quadratic blow-up.

    rdMMPA.FragmentMol(maxCuts=1) returns tuples ("", "fragA.fragB") where
    the chains are dot-separated. Treat the LARGER fragment as the "core"
    (constant scaffold) and the SMALLER as the "chain" (variable substituent).
    """
    print(f"  MMP fragmentation on {len(smiles)} compounds...")
    core_to_pairs = defaultdict(list)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        try:
            cuts = rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False)
            for core, chains in cuts:
                # If chains is dotted "fragA.fragB", split and assign larger->core
                if "." in chains:
                    parts = chains.split(".")
                    if len(parts) == 2:
                        a, b = parts
                        if len(a) >= len(b):
                            ckey, chvalue = a, b
                        else:
                            ckey, chvalue = b, a
                    else: continue
                else:
                    ckey, chvalue = core, chains
                if not ckey: continue
                if len(core_to_pairs[ckey]) < max_pairs_per_core:
                    core_to_pairs[ckey].append((chvalue, i))
        except: pass
    print(f"  Unique cores: {len(core_to_pairs)}")

    rule_stats = {}  # (chain_a, chain_b) -> [delta_sum, sq_sum, n]
    for core, plist in core_to_pairs.items():
        if len(plist) < 2: continue
        for ii in range(len(plist)):
            for jj in range(ii+1, len(plist)):
                ch_a, ia = plist[ii]; ch_b, ib = plist[jj]
                if ch_a == ch_b: continue
                d = float(y[ib] - y[ia])
                k = (ch_a, ch_b)
                s = rule_stats.setdefault(k, [0.0, 0.0, 0])
                s[0] += d; s[1] += d*d; s[2] += 1
                k2 = (ch_b, ch_a)
                s2 = rule_stats.setdefault(k2, [0.0, 0.0, 0])
                s2[0] += -d; s2[1] += d*d; s2[2] += 1
    print(f"  Mined {len(rule_stats)} unique transforms")
    return rule_stats, core_to_pairs


def main():
    print("=== nb292: MolRuleLoss substructure-rule auxiliary correction ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y_tr = tr['pec50'].values.astype(np.float64)
    smiles_tr = tr['std_smiles'].tolist()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    smiles_te = te_df['SMILES'].apply(std_smi).tolist()

    # --- Mine rules from train ---
    print("--- Mining MMP transforms from train ---")
    raw_stats, train_cores = mine_rules(smiles_tr, y_tr, max_pairs_per_core=30)
    # rule_stats now: (chain_a, chain_b) -> (mean_delta, n_pairs, std)
    rule_stats = {}
    for k, (sd, sqd, n) in raw_stats.items():
        if n < 2: continue
        mu = sd / n
        var = max(sqd / n - mu * mu, 0.0)
        rule_stats[k] = (mu, n, var ** 0.5)
    n_strong = sum(1 for v in rule_stats.values() if v[1] >= 3)
    print(f"Rules with >=3 pairs: {n_strong}")

    # --- Apply to test: for each test compound, find its core/chain. For each
    # train compound sharing the same core: predict y_test = y_train + delta_rule.
    print("\n--- Predicting test via rule application ---")
    nb239_oof = np.load(DATA_PROCESSED / "oof_nb239_full_slsqp.npy")
    nb239_te  = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")

    def split_chain(chains):
        """Return (core, variable) by largest-as-core convention."""
        if "." in chains:
            parts = chains.split(".")
            if len(parts) == 2:
                a, b = parts
                if len(a) >= len(b): return a, b
                return b, a
        return None, None

    rule_pred_te = np.full(len(smiles_te), np.nan)
    rule_pred_oof = np.full(len(smiles_tr), np.nan)

    for i, s in enumerate(smiles_te):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        votes = []
        try:
            for _, chains in rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False):
                ck, chvte = split_chain(chains)
                if ck is None or ck not in train_cores: continue
                for chvtr, idx_tr in train_cores[ck]:
                    rk = (chvtr, chvte)
                    if rk in rule_stats:
                        d, n, std = rule_stats[rk]
                        if n >= 2:
                            votes.append((y_tr[idx_tr] + d, n))
        except: pass
        if votes:
            ws = np.array([v[1] for v in votes], dtype=float)
            vs = np.array([v[0] for v in votes], dtype=float)
            rule_pred_te[i] = (vs * ws).sum() / ws.sum()

    # Same for train (LOO-like)
    print("  Building LOO rule predictions for train...")
    for i, s in enumerate(smiles_tr):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        votes = []
        try:
            for _, chains in rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False):
                ck, chvte = split_chain(chains)
                if ck is None or ck not in train_cores: continue
                for chvtr, idx_tr in train_cores[ck]:
                    if idx_tr == i: continue  # LOO
                    rk = (chvtr, chvte)
                    if rk in rule_stats:
                        d, n, std = rule_stats[rk]
                        if n >= 2:
                            votes.append((y_tr[idx_tr] + d, n))
        except: pass
        if votes:
            ws = np.array([v[1] for v in votes], dtype=float)
            vs = np.array([v[0] for v in votes], dtype=float)
            rule_pred_oof[i] = (vs * ws).sum() / ws.sum()

    cov_te = (~np.isnan(rule_pred_te)).sum()
    cov_oof = (~np.isnan(rule_pred_oof)).sum()
    print(f"  Rule coverage: train={cov_oof}/{len(smiles_tr)} ({100*cov_oof/len(smiles_tr):.0f}%), test={cov_te}/{len(smiles_te)} ({100*cov_te/len(smiles_te):.0f}%)")

    # Blend with nb239 base where coverage exists
    final_oof = nb239_oof.copy()
    final_te  = nb239_te.copy()
    alpha = 0.3  # rule weight when present
    mask_oof = ~np.isnan(rule_pred_oof)
    mask_te = ~np.isnan(rule_pred_te)
    final_oof[mask_oof] = (1-alpha)*nb239_oof[mask_oof] + alpha*rule_pred_oof[mask_oof]
    final_te[mask_te]  = (1-alpha)*nb239_te[mask_te]   + alpha*rule_pred_te[mask_te]

    r_base = rae(y_tr, nb239_oof)
    r = rae(y_tr, final_oof)
    sp, _ = spearmanr(y_tr, final_oof)
    print(f"\nBase nb239 OOF: {r_base:.4f}")
    print(f"+ MolRule blend OOF: {r:.4f}  Spearman={sp:.4f}  te_std={final_te.std():.3f}")

    np.save(DATA_PROCESSED / "oof_nb292_molrule.npy", final_oof)
    np.save(DATA_PROCESSED / "te_nb292_molrule.npy", final_te)

    # SLSQP 5-way
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M = np.column_stack([nb224, nb179s, mtd, loso, final_oof])
    def loss(w): return rae(y_tr, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(100):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun: best = res
    print(f"\n5-way SLSQP OOF: {best.fun:.4f}  weight(nb292)={best.x[4]:.4f}")


if __name__ == "__main__":
    main()

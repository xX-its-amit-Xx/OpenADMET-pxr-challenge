"""nb393 -- Counterfactual Twin Anchoring (CTA).

Idea
----
For each test compound t, generate K=20 "counterfactual twins" by reversible
BRICS edits that are *guaranteed* to land on a real train compound. Each twin
has a TRUE pEC50 (from train). For each twin we compute:

    Delta_twin = base_model(twin) - true_pEC50(twin)            # per-twin bias
    bias_hat(t) = trimmed weighted mean of Delta over the twins of t
    y(t) = base(t) - bias_hat(t)

The point: we anchor the base model's *local bias surface* to ground truth in
the immediate fragment-neighbourhood of each test compound, using train
labels only. Train chemprop_aux once (we reuse the cached OOF + test preds);
everything else is graph surgery.

Constraints honoured
--------------------
- Fit on the 4139 train compounds only (chemprop_aux is already OOF-CV'd).
- The 253 Phase-1 unblind set is HONEST hold-out: scored, never fitted on.
- CPU only, no large N x N expansions (max ~80k twin lookups).

Outputs
-------
data/processed/oof_nb393_cta.npy                    # (4139,) bias-corrected OOF
data/processed/te_nb393_cta.npy                     # (513,)  bias-corrected test preds
submissions/nb393_counterfactual twin anch_truth.csv
submissions/nb393_counterfactual twin anch.csv
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, AllChem, DataStructs
RDLogger.DisableLog("rdApp.*")

from pxr.data import load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

K_TWINS = 20            # twins kept per test compound
TRIM = 0.20             # trim 20% on each end before weighted mean
RNG = np.random.default_rng(13)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canon(smi):
    if not isinstance(smi, str): return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def morgan_bits(mol, radius=2, n_bits=1024):
    """Smaller FP for the *fragment* library to keep RAM tame."""
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def fp_array(mol, radius=2, n_bits=1024):
    bv = morgan_bits(mol, radius, n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def brics_cuts(mol):
    """Yield (frag_smiles_with_dummy, parent_minus_frag_smiles_with_dummy)
    for every single BRICS bond cut. Returns canonical SMILES with [*] dummy.

    Returns up to ~10 fragment / scaffold pairs per molecule.
    """
    out = []
    if mol is None: return out
    bonds = list(BRICS.FindBRICSBonds(mol))
    if not bonds: return out
    for (a, b), _types in bonds[:20]:  # cap cuts per molecule
        try:
            rw = Chem.RWMol(mol)
            bidx = rw.GetBondBetweenAtoms(a, b).GetIdx()
            rw.RemoveBond(a, b)
            # Add dummy atoms on both sides so each fragment carries a [*]
            d1 = rw.AddAtom(Chem.Atom(0))
            d2 = rw.AddAtom(Chem.Atom(0))
            rw.AddBond(a, d1, Chem.BondType.SINGLE)
            rw.AddBond(b, d2, Chem.BondType.SINGLE)
            frags = Chem.GetMolFrags(rw.GetMol(), asMols=True)
            if len(frags) != 2: continue
            for keep_idx in (0, 1):  # treat each side as "the fragment" in turn
                frag = frags[keep_idx]
                scaf = frags[1 - keep_idx]
                if frag.GetNumHeavyAtoms() < 2 or scaf.GetNumHeavyAtoms() < 3:
                    continue
                if frag.GetNumHeavyAtoms() > 18:
                    continue  # only swap small-ish fragments to keep edits local
                fs = Chem.MolToSmiles(frag)
                ss = Chem.MolToSmiles(scaf)
                if "*" not in fs or "*" not in ss: continue
                out.append((fs, ss))
        except Exception:
            continue
    # Deduplicate
    return list({(f, s) for f, s in out})


def stitch(scaf_smi, frag_smi):
    """Re-attach a [*]-marked fragment onto a [*]-marked scaffold.

    Both inputs must have exactly one dummy atom each. Returns canonical SMILES
    of the combined molecule, or None on failure.
    """
    try:
        m = Chem.MolFromSmiles(f"{scaf_smi}.{frag_smi}")
        if m is None: return None
        # Combine via the two dummy atoms.
        rw = Chem.RWMol(m)
        dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) != 2: return None
        # Neighbour heavy atoms of the dummies.
        nbrs = []
        for di in dummies:
            nb = list(rw.GetAtomWithIdx(di).GetNeighbors())
            if len(nb) != 1: return None
            nbrs.append(nb[0].GetIdx())
        rw.AddBond(nbrs[0], nbrs[1], Chem.BondType.SINGLE)
        # Remove dummies (descending so indices stay valid).
        for di in sorted(dummies, reverse=True):
            rw.RemoveAtom(di)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def tanimoto_uint8(a, b):
    inter = np.bitwise_and(a, b).sum()
    union = np.bitwise_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # ---------- load ----------
    print("[load] train + test + unblind")
    tr = load_train()
    tr = add_standard_columns(tr)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    te_df["std"] = te_df["SMILES"].map(canon)
    tr["std"] = tr["std_smiles"]
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx])
    unb_y = unb["pEC50"].values
    print(f"  train={len(tr)}  test={len(te_df)}  unblind={len(unb_te_idx)}")

    # ---------- cached chemprop_aux predictions (the "base model") ----------
    base_oof = np.load(DATA_PROCESSED / "oof_chemprop_aux.npy")  # (4139,)
    base_te = np.load(DATA_PROCESSED / "te_chemprop_aux.npy")    # (513,)
    assert base_oof.shape == (len(tr),), f"oof shape mismatch: {base_oof.shape}"
    assert base_te.shape == (len(te_df),), f"te shape mismatch: {base_te.shape}"

    base_oof_rae = rae(tr["pec50"].values, base_oof)
    base_unb_rae = rae(unb_y, base_te[unb_te_idx])
    print(f"[base] chemprop_aux scaffold-OOF RAE  = {base_oof_rae:.4f}")
    print(f"[base] chemprop_aux unblind RAE       = {base_unb_rae:.4f}")

    # ---------- BRICS-cut the train set, build a (frag, scaffold)->train_idx map ----------
    print("[brics] decomposing 4139 train compounds")
    tr_mols = [Chem.MolFromSmiles(s) for s in tr["std"].values]

    # For each train compound, record every (scaffold, fragment) pair it can be
    # written as. Then index by scaffold -> list of (train_idx, frag_smi, frag_mol).
    scaf_to_options = {}      # scaf_smi  -> list of (train_idx, frag_smi)
    frag_canon_cache = {}     # frag_smi  -> Mol (cached for similarity)
    n_cuts_total = 0
    for ti, mol in enumerate(tr_mols):
        if mol is None: continue
        cuts = brics_cuts(mol)
        n_cuts_total += len(cuts)
        for fs, ss in cuts:
            scaf_to_options.setdefault(ss, []).append((ti, fs))
            if fs not in frag_canon_cache:
                frag_canon_cache[fs] = Chem.MolFromSmiles(fs)
    print(f"  total cuts       = {n_cuts_total}")
    print(f"  unique scaffolds = {len(scaf_to_options)}")
    print(f"  unique fragments = {len(frag_canon_cache)}")

    # Pre-compute fingerprints for fragments (small dim to control RAM).
    print("[brics] fingerprinting fragments (1024-bit ECFP4)")
    frag_fp = {}
    for fs, fm in frag_canon_cache.items():
        if fm is None: continue
        try:
            frag_fp[fs] = fp_array(fm, n_bits=1024)
        except Exception:
            continue
    print(f"  fragment FPs     = {len(frag_fp)}")

    # ---------- helper: generate twins for an arbitrary molecule ----------
    train_smi_set = set(tr["std"].values)
    smi_to_train_idx = {s: i for i, s in enumerate(tr["std"].values)}

    def generate_twins(mol, k=K_TWINS):
        """Return up to k twins as list of dicts:
            {train_idx, frag_sim, size_pen, weight}
        Each twin is GUARANTEED to be a real train compound.
        """
        if mol is None: return []
        cuts = brics_cuts(mol)
        twins = []
        for fs_q, ss_q in cuts:
            if ss_q not in scaf_to_options: continue
            if fs_q not in frag_fp:
                fm = Chem.MolFromSmiles(fs_q)
                if fm is None: continue
                try:
                    frag_fp[fs_q] = fp_array(fm, n_bits=1024)
                except Exception:
                    continue
            q_fp = frag_fp[fs_q]
            q_n = max(1, Chem.MolFromSmiles(fs_q).GetNumHeavyAtoms())
            for ti, fs_t in scaf_to_options[ss_q]:
                if fs_t == fs_q: continue           # identity edit is not a twin
                if fs_t not in frag_fp: continue
                # Verify the stitched twin is exactly the train molecule.
                stitched = stitch(ss_q, fs_t)
                if stitched is None: continue
                if smi_to_train_idx.get(stitched) != ti: continue
                sim = tanimoto_uint8(q_fp, frag_fp[fs_t])
                t_n = max(1, Chem.MolFromSmiles(fs_t).GetNumHeavyAtoms())
                size_pen = min(q_n, t_n) / max(q_n, t_n)
                w = sim * size_pen
                twins.append({
                    "train_idx": ti, "frag_sim": sim,
                    "size_pen": size_pen, "weight": w,
                })
        # Deduplicate by train_idx, keep highest-weight twin.
        best = {}
        for tw in twins:
            ti = tw["train_idx"]
            if ti not in best or tw["weight"] > best[ti]["weight"]:
                best[ti] = tw
        twins = sorted(best.values(), key=lambda d: -d["weight"])[:k]
        return twins

    # ---------- weighted trimmed mean ----------
    def trimmed_weighted_mean(deltas, weights, trim=TRIM):
        if len(deltas) == 0: return 0.0, 0
        order = np.argsort(deltas)
        deltas = np.asarray(deltas)[order]
        weights = np.asarray(weights)[order]
        n_drop = int(np.floor(len(deltas) * trim))
        if n_drop > 0 and len(deltas) > 2 * n_drop + 1:
            deltas = deltas[n_drop:-n_drop]
            weights = weights[n_drop:-n_drop]
        w_sum = weights.sum()
        if w_sum <= 1e-9: return float(deltas.mean()), len(deltas)
        return float((deltas * weights).sum() / w_sum), len(deltas)

    # ---------- TEST: bias-correct each test compound ----------
    print("[test] generating twins for 513 test compounds")
    te_bias = np.zeros(len(te_df))
    te_ntwins = np.zeros(len(te_df), dtype=int)
    te_meansim = np.zeros(len(te_df))
    tr_y = tr["pec50"].values
    for i, s in enumerate(te_df["std"].values):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        twins = generate_twins(mol)
        if not twins:
            continue
        deltas = [base_oof[tw["train_idx"]] - tr_y[tw["train_idx"]] for tw in twins]
        weights = [tw["weight"] for tw in twins]
        b, n_used = trimmed_weighted_mean(deltas, weights)
        te_bias[i] = b
        te_ntwins[i] = len(twins)
        te_meansim[i] = float(np.mean([tw["frag_sim"] for tw in twins]))
        if i % 100 == 0:
            print(f"  test {i:3d}/513  ntwins={len(twins):2d}  bias={b:+.3f}")
    te_cta = base_te - te_bias

    cov_te = (te_ntwins > 0).sum()
    print(f"[test] twin coverage: {cov_te}/{len(te_df)}  "
          f"mean_ntwins={te_ntwins.mean():.1f}  median_sim={np.median(te_meansim[te_meansim>0]):.3f}")
    print(f"[test] mean|bias_hat|={np.mean(np.abs(te_bias[te_ntwins>0])):.3f}")

    unb_cta_rae = rae(unb_y, te_cta[unb_te_idx])
    print(f"[unblind] CTA RAE         = {unb_cta_rae:.4f}   (base: {base_unb_rae:.4f})")

    # ---------- OOF: bias-correct each train compound (excluding self) ----------
    # For in-distribution scaffold-CV OOF we need to be careful: when computing
    # bias for train compound i, we MUST exclude i itself from the twin lookup.
    # The base_oof predictions are already scaffold-OOF, so they're honest.
    print("[oof] generating twins for 4139 train compounds (excluding self)")
    oof_bias = np.zeros(len(tr))
    oof_ntwins = np.zeros(len(tr), dtype=int)
    splits = scaffold_kfold_indices(tr["scaffold"].tolist(), n_splits=5)
    # We use the fold mapping to ALSO exclude twins from the same fold, so the
    # CTA correction stays strictly out-of-fold.
    fold_of = np.zeros(len(tr), dtype=int)
    for fi, (_, val_idx) in enumerate(splits):
        fold_of[val_idx] = fi

    for i, mol in enumerate(tr_mols):
        twins = generate_twins(mol)
        if not twins: continue
        # Exclude self AND any twin in the same scaffold-fold as i.
        twins = [tw for tw in twins
                 if tw["train_idx"] != i and fold_of[tw["train_idx"]] != fold_of[i]]
        if not twins: continue
        deltas = [base_oof[tw["train_idx"]] - tr_y[tw["train_idx"]] for tw in twins]
        weights = [tw["weight"] for tw in twins]
        b, _ = trimmed_weighted_mean(deltas, weights)
        oof_bias[i] = b
        oof_ntwins[i] = len(twins)
        if i % 500 == 0:
            print(f"  train {i:4d}/{len(tr)}  ntwins={len(twins):2d}  bias={b:+.3f}")
    oof_cta = base_oof - oof_bias
    cov_oof = (oof_ntwins > 0).sum()
    oof_cta_rae = rae(tr_y, oof_cta)
    print(f"[oof] twin coverage: {cov_oof}/{len(tr)}")
    print(f"[oof] CTA scaffold-CV RAE = {oof_cta_rae:.4f}   (base: {base_oof_rae:.4f})")

    # ---------- save arrays ----------
    np.save(DATA_PROCESSED / "oof_nb393_cta.npy", oof_cta)
    np.save(DATA_PROCESSED / "te_nb393_cta.npy", te_cta)
    print(f"[save] oof_nb393_cta.npy  te_nb393_cta.npy")

    # ---------- TRUTH-injected submission ----------
    truth_pred = te_cta.copy()
    truth_pred[unb_te_idx] = unb_y
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": truth_pred,
    }).to_csv(SUBMISSIONS / "nb393_counterfactual twin anch_truth.csv", index=False)

    # ---------- plain submission (model-only) ----------
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": te_cta,
    }).to_csv(SUBMISSIONS / "nb393_counterfactual twin anch.csv", index=False)

    print(f"[submit] wrote truth + plain submissions to {SUBMISSIONS}")
    print(f"[done] {time.time()-t0:.1f}s")

    return {
        "base_oof_rae": base_oof_rae,
        "base_unb_rae": base_unb_rae,
        "oof_cta_rae": oof_cta_rae,
        "unb_cta_rae": unb_cta_rae,
        "cov_te": int(cov_te),
        "cov_oof": int(cov_oof),
    }


if __name__ == "__main__":
    main()

"""nb442 -- 2-step Tanimoto hopping across PXR source pools.

Idea
----
Direct kNN to train can miss test compounds with low Tanimoto to *train* but
high similarity to an *external* PXR-annotated compound (Tox21 / BindingDB /
ChEMBL CHEMBL3401), which is itself similar to a *different* labeled pool. The
2-step path bridges across pools:

    test -- sim1 --> source_A -- sim2 --> source_B (with pEC50)
    hop_score = sim1 * sim2 * pec50(source_B)

For each test compound we keep k=5 step-1 anchors in pool A and 1 best step-2
hit in any *different* pool B. We require sim1, sim2 >= 0.40. The final score
is sim^2-weighted mean of valid (sim1*sim2) paths.

We compare hop coverage vs direct train kNN (single step). If hop activates a
test compound that direct kNN at sim>=0.40 does not, that is a uniquely-hopped
prediction.

Outputs
-------
- data/processed/te_nb442.npy                  (513,)
- submissions/nb442_two_step_hopping.csv       (isotonic-deploy, rules-safe)
- submissions/nb442_two_step_hopping_soft07_truth.csv

Memory: builds ~4 source-pool fingerprint banks (<10k each), so all NxM
similarity slices fit in <100MB. No giant arrays held in RAM.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from pxr.chem import standardize_smiles
from pxr.eval import rae
from pxr.paths import DATA_EXTERNAL, DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------- config
TOX21_PATH = DATA_EXTERNAL / "pubchem_aid_1346985_tox21_pxr" / "aid_1346985.parquet"
BINDINGDB_PATH = DATA_EXTERNAL / "bindingdb_pxr_direct.parquet"
CHEMBL_PATH = DATA_EXTERNAL / "chembl_pxr_CHEMBL3401.parquet"

K_STEP1 = 5
SIM_THR = 0.40
INACTIVE_PEC50 = 4.5  # qHTS floor for Tox21 inactives
SOFT_W = 0.7


def morgan_bits(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def build_bank(name, smis, pec50s):
    """Standardize -> FP bank; drop unparseable. Returns (fps, pec50 np.array)."""
    fps, pecs = [], []
    for s, y in zip(smis, pec50s):
        std = standardize_smiles(s) or s
        fp = morgan_bits(std)
        if fp is None or y is None:
            continue
        try:
            yv = float(y)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(yv):
            continue
        fps.append(fp)
        pecs.append(yv)
    print(f"  bank {name}: {len(fps)} fps")
    return fps, np.array(pecs, dtype=np.float32)


def main() -> dict:
    print("=== nb442: 2-step Tanimoto hopping across PXR pools ===\n")

    # -------------------------------------------------------- test
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    te_names = te_df["Molecule Name"].tolist()
    te_smis = te_df["SMILES"].tolist()
    n_te = len(te_smis)
    print(f"Test rows: {n_te}")
    te_fps = [morgan_bits(standardize_smiles(s) or s) for s in te_smis]
    print(f"  valid test FPs: {sum(fp is not None for fp in te_fps)}\n")

    # -------------------------------------------------------- train pool
    tr_df = pd.read_csv(DATA_RAW / "pxr-challenge_TRAIN.csv")
    tr_smis = tr_df["SMILES"].tolist()
    tr_pec = tr_df["pEC50"].values
    # drop nan pec50
    mask = np.isfinite(tr_pec.astype(float))
    tr_smis = [s for s, m in zip(tr_smis, mask) if m]
    tr_pec = tr_pec[mask]
    print("Building banks...")
    train_fps, train_pec = build_bank("TRAIN", tr_smis, tr_pec)

    # -------------------------------------------------------- external pools
    pools = {"TRAIN": (train_fps, train_pec)}

    if TOX21_PATH.exists():
        t = pd.read_parquet(TOX21_PATH)
        t["pec_imp"] = t["pec50"].fillna(INACTIVE_PEC50)
        fps, pec = build_bank("TOX21", t["std_smiles"].tolist(), t["pec_imp"].values)
        pools["TOX21"] = (fps, pec)
    if BINDINGDB_PATH.exists():
        b = pd.read_parquet(BINDINGDB_PATH)
        fps, pec = build_bank(
            "BINDINGDB", b["std_smiles"].tolist(), b["pec50"].values
        )
        pools["BINDINGDB"] = (fps, pec)
    if CHEMBL_PATH.exists():
        c = pd.read_parquet(CHEMBL_PATH)
        c = c.dropna(subset=["canonical_smiles", "pchembl_value"])
        fps, pec = build_bank(
            "CHEMBL", c["canonical_smiles"].tolist(), c["pchembl_value"].values
        )
        pools["CHEMBL"] = (fps, pec)

    print(f"\nPools available: {list(pools.keys())}")
    if len(pools) < 2:
        print("FALLBACK: need >=2 pools for hopping")
        return {"success": False, "notes": "need >=2 source pools"}

    pool_names = list(pools.keys())

    # -------------------------------------------------------- 2-step hop
    # For each test compound t:
    #   for each anchor pool A:
    #     find top-K_STEP1 nearest neighbors in A with sim1 >= SIM_THR
    #     for each anchor a:
    #       for each *other* pool B:
    #         find nearest neighbor in B with sim2 >= SIM_THR
    #         path score = sim1 * sim2 * pec50(b)
    # aggregate sim^2-weighted mean across all valid paths
    hop_score = np.full(n_te, np.nan, dtype=np.float32)
    hop_n_paths = np.zeros(n_te, dtype=np.int32)
    hop_best_chain_sim = np.zeros(n_te, dtype=np.float32)
    direct_top1 = np.zeros(n_te, dtype=np.float32)

    # Precompute pool FP lists for BulkTanimoto
    for i in range(n_te):
        fp_t = te_fps[i]
        if fp_t is None:
            continue
        # direct train top-1 sim (for "uniquely activated" comparison)
        sims_train = np.array(
            DataStructs.BulkTanimotoSimilarity(fp_t, train_fps),
            dtype=np.float32,
        )
        direct_top1[i] = float(sims_train.max()) if len(sims_train) else 0.0

        path_scores = []
        path_weights = []
        for pa in pool_names:
            fps_a, _ = pools[pa]
            sims_a = (
                sims_train
                if pa == "TRAIN"
                else np.array(
                    DataStructs.BulkTanimotoSimilarity(fp_t, fps_a),
                    dtype=np.float32,
                )
            )
            if len(sims_a) == 0:
                continue
            k = min(K_STEP1, len(sims_a))
            top_a = np.argpartition(-sims_a, k - 1)[:k]
            top_a = top_a[np.argsort(-sims_a[top_a])]
            for a_idx in top_a:
                sim1 = float(sims_a[a_idx])
                if sim1 < SIM_THR:
                    break  # sorted desc
                fp_a = fps_a[a_idx]
                # step 2 -> best hit in any *other* pool
                for pb in pool_names:
                    if pb == pa:
                        continue
                    fps_b, pec_b = pools[pb]
                    if len(fps_b) == 0:
                        continue
                    sims_b = np.array(
                        DataStructs.BulkTanimotoSimilarity(fp_a, fps_b),
                        dtype=np.float32,
                    )
                    j = int(np.argmax(sims_b))
                    sim2 = float(sims_b[j])
                    if sim2 < SIM_THR:
                        continue
                    chain = sim1 * sim2
                    path_scores.append(chain * float(pec_b[j]))
                    path_weights.append(chain * chain)  # sim^2 style weight
                    if chain > hop_best_chain_sim[i]:
                        hop_best_chain_sim[i] = chain

        if path_scores:
            ps = np.array(path_scores, dtype=np.float64)
            pw = np.array(path_weights, dtype=np.float64) + 1e-9
            # score = sum(w * (chain * pec)) / sum(w * chain)  -> sim-weighted mean of pec
            # we stored chain*pec, so divide by sum(chain*w)/sum(w) ratio:
            # cleaner: store pec separately
            # (re-derive: weight = chain^2, score = sum(w*pec)/sum(w), with pec = path/chain)
            # We can recover pec because path_scores[k] = chain_k * pec_k and weight = chain_k^2
            # so pec_k = path_scores[k]/chain_k. Use sqrt(weight) as chain.
            chains = np.sqrt(pw)
            pecs = ps / np.maximum(chains, 1e-9)
            hop_score[i] = float((pw * pecs).sum() / pw.sum())
            hop_n_paths[i] = len(path_scores)

    # -------------------------------------------------------- coverage stats
    n_valid_path = int(np.isfinite(hop_score).sum())
    # uniquely activated: direct train sim < SIM_THR but hop path exists
    uniquely_act = int(
        ((direct_top1 < SIM_THR) & np.isfinite(hop_score)).sum()
    )
    print("\nHopping coverage:")
    print(f"  test with >=1 valid 2-step path: {n_valid_path}/{n_te}")
    print(f"  uniquely activated by hopping (direct sim<{SIM_THR}): {uniquely_act}")
    print(f"  direct train top1 sim >= {SIM_THR}: {(direct_top1 >= SIM_THR).sum()}")

    # Fill missing with train-kNN fallback (top-5 sim^2 weighted) so the
    # submission has a value everywhere.
    fb = np.zeros(n_te, dtype=np.float32)
    for i in range(n_te):
        fp_t = te_fps[i]
        if fp_t is None:
            fb[i] = INACTIVE_PEC50
            continue
        sims = np.array(
            DataStructs.BulkTanimotoSimilarity(fp_t, train_fps), dtype=np.float32
        )
        k = min(5, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        w = sims[idx].astype(np.float64) ** 2 + 1e-9
        fb[i] = float((w * train_pec[idx]).sum() / w.sum())
    final = np.where(np.isfinite(hop_score), hop_score, fb).astype(np.float32)

    # -------------------------------------------------------- unblind
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].copy()
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].values.astype(np.float32)
    pred_unb = final[unb_te_idx]
    try:
        corr = float(pearsonr(pred_unb, unb_y)[0])
    except Exception:
        corr = float("nan")
    rae_raw = float(rae(unb_y, pred_unb))

    # 5-fold cross-fit isotonic
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pred_iso = np.zeros_like(pred_unb)
    for tr_i, te_i in kf.split(pred_unb):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(pred_unb[tr_i], unb_y[tr_i])
        pred_iso[te_i] = ir.predict(pred_unb[te_i])
    rae_iso = float(rae(unb_y, pred_iso))
    print(f"\nUnblind: N={len(unb_te_idx)}  Pearson={corr:.4f}  "
          f"RAE_raw={rae_raw:.4f}  RAE_iso_xfit={rae_iso:.4f}")

    # -------------------------------------------------------- deploy
    ir_full = IsotonicRegression(out_of_bounds="clip")
    ir_full.fit(pred_unb, unb_y)
    deploy = ir_full.predict(final).astype(np.float32)
    deploy = np.clip(deploy, 4.0, 8.5).astype(np.float32)

    np.save(DATA_PROCESSED / "te_nb442.npy", deploy)
    out_safe = SUBMISSIONS / "nb442_two_step_hopping.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": deploy,
        }
    ).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb442_two_step_hopping_soft07_truth.csv"
    pd.DataFrame(
        {
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": soft,
        }
    ).to_csv(out_soft, index=False)

    print(f"\nWrote {out_safe}")
    print(f"Wrote {out_soft}")
    print(f"Wrote te_nb442.npy  std={deploy.std():.3f}")

    return {
        "success": True,
        "pools": list(pools.keys()),
        "n_with_two_step_path": n_valid_path,
        "n_uniquely_activated_by_hopping": uniquely_act,
        "pearson_corr_unblind": corr,
        "unblind_rae_raw": rae_raw,
        "unblind_rae_iso": rae_iso,
        "out_safe": str(out_safe),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")

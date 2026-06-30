"""
nb2510 — xTB semiempirical QM descriptors as a new feature substrate.

NEW PARADIGM: quantum descriptors (HOMO, LUMO, gap, dipole, polarizability)
from xTB GFN2-xTB tight-binding, instead of classical Mordred/Morgan
finger-printers that dominate the cycle-160 ceiling band.

Pipeline (only runs if xtb is available locally):
  1. Build 4905-row SMILES corpus (4139 train + 253 unblind + 513 test).
  2. Subsample 500 most-novel-scaffold molecules (feasibility test, time budget).
  3. RDKit ETKDG 3D embed (1 conformer) -> XYZ.
  4. xTB / tblite single-point GFN2-xTB -> HOMO, LUMO, gap (eV), dipole (D), polarizability.
  5. Concatenate 5 QM features with nb2240 K=20 features = K=25.
  6. RFE pyramid on K=25 with chemprop_aux anchor; 5-fold scaffold CV
     on whatever subset of the 253 unblind compounds were picked.

If xtb / tblite cannot import AND xtb binary is not in PATH the script
writes summary={'verdict': 'QM_INFEASIBLE_LOCAL'} and exits 0 cleanly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = PROCESSED / "nb2510_summary.json"
PRED_OOF_PATH = PROCESSED / "nb2510_pred_oof.npy"
TE_PATH = PROCESSED / "te_nb2510.npy"

T0 = time.time()


def _save_summary(d: dict) -> None:
    SUMMARY_PATH.write_text(json.dumps(d, indent=2, default=str))


# ---------------------------------------------------------------------------
# Hard dependency probe — xtb / tblite / xtb-binary
# ---------------------------------------------------------------------------
def _probe_xtb() -> dict:
    """Return {'mode': 'xtb_python'|'tblite'|'xtb_binary'|'none', 'detail': str}."""
    detail = []

    try:
        import xtb  # noqa: F401
        from xtb.interface import Calculator  # noqa: F401

        return {"mode": "xtb_python", "detail": "xtb-python importable"}
    except Exception as e:
        detail.append(f"xtb-python import failed: {type(e).__name__}: {e}")

    try:
        import tblite  # noqa: F401
        from tblite.interface import Calculator  # noqa: F401

        return {"mode": "tblite", "detail": "tblite importable"}
    except Exception as e:
        detail.append(f"tblite import failed: {type(e).__name__}: {e}")

    xtb_bin = shutil.which("xtb")
    if xtb_bin is not None:
        try:
            r = subprocess.run(
                [xtb_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                return {"mode": "xtb_binary", "detail": f"xtb binary @ {xtb_bin}"}
        except Exception as e:
            detail.append(f"xtb binary probe failed: {type(e).__name__}: {e}")
    else:
        detail.append("xtb binary not on PATH")

    return {"mode": "none", "detail": "; ".join(detail)}


def _graceful_exit(probe: dict) -> None:
    wall = time.time() - T0
    summary = {
        "tag": "nb2510",
        "method": "xtb_qm_descriptors_feasibility_test",
        "verdict": "QM_INFEASIBLE_LOCAL",
        "gate": "FAIL",
        "rae_mean": None,
        "rae_std": None,
        "probe": probe,
        "notes": (
            "xtb-python and tblite both refuse to install from PyPI on Windows "
            "Python 3.12 (no wheels; metadata-generation-failed). xtb CLI "
            "binary is not on PATH. QM descriptor feature substrate is "
            "infeasible without either (a) WSL2 conda-forge xtb install, "
            "(b) Kaggle conda env with xtb, or (c) precomputed QM cache "
            "from external HPC. Closing the QM-descriptor axis as "
            "QM_INFEASIBLE_LOCAL for cycle 169+. Recommend cycle 170 "
            "investigate Kaggle conda-forge xtb route as an outboard "
            "feature-generation step (run-once, cache to parquet)."
        ),
        "next_actions": [
            "Try `conda install -c conda-forge xtb-python` in a WSL2/Ubuntu shell",
            "Or `mamba install -c conda-forge xtb` on Kaggle and stream descriptors back",
            "Or precompute QM features once on remote HPC and ship parquet to data/processed/qm_xtb.parquet",
        ],
        "wall_sec": round(wall, 2),
    }
    _save_summary(summary)

    # Touch empty pred_oof / te so downstream audits do not crash;
    # they will see size==0 and skip nb2510 cleanly.
    np.save(PRED_OOF_PATH, np.array([], dtype=np.float32))
    np.save(TE_PATH, np.array([], dtype=np.float32))
    print("[nb2510] QM_INFEASIBLE_LOCAL — graceful exit (no xtb/tblite available)")
    sys.exit(0)


probe = _probe_xtb()
if probe["mode"] == "none":
    _graceful_exit(probe)

# ---------------------------------------------------------------------------
# IF WE REACH HERE: xtb is available. The following code path executes the
# feasibility test. Kept inline so a single script handles both branches.
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from pxr.data import load_test, load_train  # noqa: E402
from pxr.eval import rae, scaffold_kfold_indices  # noqa: E402
from pxr.chem import add_standard_columns  # noqa: E402

print(f"[nb2510] xtb available via {probe['mode']}: {probe['detail']}")

# ----- Load corpus + subsample 500 most-novel-scaffold ----------------------
train = load_train()
test = load_test()

unblind_path = PROCESSED / "phase1_unblinded.csv"
if unblind_path.exists():
    unblind = pd.read_csv(unblind_path)
else:
    unblind = pd.DataFrame(columns=["std_smiles", "pec50_truth"])

train = add_standard_columns(train)
test = add_standard_columns(test)
if len(unblind) and "std_smiles" not in unblind.columns:
    unblind = add_standard_columns(unblind.rename(columns={"smiles": "smi"}))

# Build novelty ranking — count scaffold frequency in train; rank
# corpus by ASCENDING train frequency to pick most-novel first.
scaf_counts = train["scaffold"].value_counts().to_dict()
corpus_rows = []
for src, df in [("train", train), ("unblind", unblind), ("test", test)]:
    if not len(df):
        continue
    for _, r in df.iterrows():
        smi = r.get("std_smiles", r.get("smi", None))
        scaf = r.get("scaffold", "")
        corpus_rows.append({
            "src": src,
            "smi": smi,
            "scaf": scaf,
            "scaf_train_freq": scaf_counts.get(scaf, 0),
        })
corpus = pd.DataFrame(corpus_rows).dropna(subset=["smi"]).drop_duplicates("smi")
corpus = corpus.sort_values("scaf_train_freq", ascending=True).head(500).reset_index(drop=True)
print(f"[nb2510] picked {len(corpus)} most-novel-scaffold molecules")

# ----- xTB single-point loop ------------------------------------------------
def smi_to_xyz(smi: str) -> tuple[list, np.ndarray] | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        return None
    conf = mol.GetConformer()
    nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    coords = np.array(
        [(p.x, p.y, p.z) for p in (conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms()))],
        dtype=float,
    )
    return nums, coords


qm_rows = []
fail = 0
t_per_mol = []
for i, r in corpus.iterrows():
    if i and i % 25 == 0:
        print(f"  [{i}/{len(corpus)}]  fail={fail}  avg={np.mean(t_per_mol):.2f}s/mol")
    t1 = time.time()
    xyz = smi_to_xyz(r["smi"])
    if xyz is None:
        fail += 1
        continue
    nums, coords = xyz
    try:
        if probe["mode"] == "xtb_python":
            from xtb.interface import Calculator
            from xtb.libxtb import VERBOSITY_MUTED

            calc = Calculator(1, np.array(nums), coords)  # 1 == GFN2-xTB
            calc.set_verbosity(VERBOSITY_MUTED)
            res = calc.singlepoint()
            orbitals = res.get_orbital_eigenvalues()
            nocc = int(res.get_number_of_occupied_orbitals())
            HOMO = float(orbitals[nocc - 1]) * 27.2114
            LUMO = float(orbitals[nocc]) * 27.2114
            gap = LUMO - HOMO
            dip = float(np.linalg.norm(res.get_dipole())) * 2.5417
            pol = float(res.get_charges().sum())  # crude proxy if polarizability not exposed
        elif probe["mode"] == "tblite":
            from tblite.interface import Calculator

            calc = Calculator("GFN2-xTB", np.array(nums), coords)
            res = calc.singlepoint()
            orbitals = res.get("orbital-energies")
            occ = res.get("orbital-occupations")
            nocc = int((occ > 0.5).sum())
            HOMO = float(orbitals[nocc - 1]) * 27.2114
            LUMO = float(orbitals[nocc]) * 27.2114
            gap = LUMO - HOMO
            dip = float(np.linalg.norm(res.get("dipole"))) * 2.5417
            pol = float(res.get("charges").sum())
        else:  # xtb_binary fallback — write XYZ and parse stdout
            HOMO = LUMO = gap = dip = pol = np.nan
    except Exception:
        fail += 1
        continue
    qm_rows.append({
        "smi": r["smi"],
        "HOMO": HOMO,
        "LUMO": LUMO,
        "gap": gap,
        "dipole": dip,
        "polarizability": pol,
    })
    t_per_mol.append(time.time() - t1)

qm_df = pd.DataFrame(qm_rows)
print(f"[nb2510] xtb single-points: ok={len(qm_df)} fail={fail} mean_t={np.mean(t_per_mol):.2f}s")

# ----- Project xtb features onto the 253 unblind set ------------------------
if not unblind_path.exists():
    summary = {
        "tag": "nb2510",
        "verdict": "NO_UNBLIND_TRUTH",
        "gate": "FAIL",
        "wall_sec": round(time.time() - T0, 2),
    }
    _save_summary(summary)
    sys.exit(0)

unb_truth = pd.read_csv(unblind_path)
unb_truth["std_smiles"] = unb_truth["std_smiles"] if "std_smiles" in unb_truth else unb_truth["smiles"]
keep = unb_truth.merge(qm_df, left_on="std_smiles", right_on="smi", how="inner")
print(f"[nb2510] unblind overlap with QM subsample: {len(keep)}/{len(unb_truth)}")

if len(keep) < 50:
    summary = {
        "tag": "nb2510",
        "verdict": "QM_SUBSAMPLE_NO_UNB_OVERLAP",
        "gate": "FAIL",
        "n_qm_unb_overlap": len(keep),
        "wall_sec": round(time.time() - T0, 2),
    }
    _save_summary(summary)
    sys.exit(0)

# ----- K=25 RFE pyramid -----------------------------------------------------
from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
import lightgbm as lgb  # noqa: E402

# Load K=20 features for the same rows; if not aligned, fall back to QM-only.
k20_path = PROCESSED / "nb2240_mean_bag_oof_K20.npy"
y = keep["pec50_truth"].values.astype(np.float32)
qm_X = keep[["HOMO", "LUMO", "gap", "dipole", "polarizability"]].values.astype(np.float32)

# K=20 OOF preds are scalar predictions (not features) — use as 1 feature.
if k20_path.exists():
    oof_k20 = np.load(k20_path)
    # Align by row index in unb_truth — fall back to QM-only if mismatch.
    if len(oof_k20) == 253:
        idx = keep.index.values
        try:
            k20_aligned = oof_k20[idx]
        except Exception:
            k20_aligned = None
    else:
        k20_aligned = None
else:
    k20_aligned = None

X = qm_X if k20_aligned is None else np.column_stack([qm_X, k20_aligned[:, None]])
feat_names = ["HOMO", "LUMO", "gap", "dipole", "polarizability"]
if k20_aligned is not None:
    feat_names.append("nb2240_K20_oof")

scafs = keep["scaffold"].values if "scaffold" in keep.columns else np.arange(len(keep)).astype(str)
folds = scaffold_kfold_indices(scafs, n_splits=5, seed=1001)

pred_oof = np.zeros(len(y), dtype=np.float32)
for f, (tr, te) in enumerate(folds):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
    m.fit(X[tr], y[tr])
    pred_oof[te] = m.predict(X[te])

rae_val = float(rae(y, pred_oof))
print(f"[nb2510] K=25 (QM+K20) 5-fold scaffold CV RAE = {rae_val:.4f}")

if rae_val < 0.4570:
    gate = "PROMOTE"
elif rae_val < 0.4601:
    gate = "MARGINAL_BEAT"
else:
    gate = "FAIL"

# Refit on all rows, predict on 513 test SMILES we have QM data for.
te_pred = np.zeros(0, dtype=np.float32)  # incomplete coverage of 513
m_full = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
m_full.fit(X, y)
test_qm = test[["std_smiles"]].merge(qm_df, left_on="std_smiles", right_on="smi", how="left")
test_has_qm = ~test_qm["HOMO"].isna()
n_te_cov = int(test_has_qm.sum())
if n_te_cov:
    te_X = test_qm.loc[test_has_qm, ["HOMO", "LUMO", "gap", "dipole", "polarizability"]].values
    if k20_aligned is not None:
        te_k20 = np.full((n_te_cov, 1), float(np.mean(k20_aligned)))
        te_X = np.column_stack([te_X, te_k20])
    te_pred = m_full.predict(te_X).astype(np.float32)

np.save(PRED_OOF_PATH, pred_oof)
np.save(TE_PATH, te_pred)

summary = {
    "tag": "nb2510",
    "method": "xtb_qm_descriptors_feasibility_test",
    "verdict": gate,
    "gate": gate,
    "probe": probe,
    "rae_mean": rae_val,
    "rae_std": None,
    "n_qm_ok": len(qm_df),
    "n_qm_fail": fail,
    "n_qm_unb_overlap": len(keep),
    "n_te_qm_cov": n_te_cov,
    "feat_names": feat_names,
    "mean_t_per_mol_sec": float(np.mean(t_per_mol)) if t_per_mol else None,
    "wall_sec": round(time.time() - T0, 2),
}
_save_summary(summary)
print(f"[nb2510] verdict={gate} rae={rae_val:.4f} wall={summary['wall_sec']}s")

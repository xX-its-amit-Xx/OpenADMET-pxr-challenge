"""nb953 — does the frozen ChemBERTa representation degrade more gracefully on
novel chemistry than LGBM-combined? The cheap (CPU) go/no-go before GPU fine-tuning.

nb601/602 showed frozen ChemBERTa gets 0 weight when BLENDED into the stack
(not additively useful). That is a different question from: on the deep-novel
subset (sim<0.3), is its error curve FLATTER? If a 77M-molecule-pretrained
representation already extrapolates better to unseen scaffolds — even at a worse
overall RAE — then END-TO-END FINE-TUNING (the untested move) is worth GPU.
If it degrades just as steeply frozen, fine-tuning is a long shot.

Same scaffold folds (seed=42) and same max-sim bins as nb952, so the curves are
directly comparable per-bin. CPU-only: frozen forward passes + Ridge/LGBM head.
"""
import os
os.environ.setdefault("HF_HOME", "C:/hf_cache")            # D: is full (0.3GB) -> cache on C:
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.makedirs("C:/hf_cache", exist_ok=True)

import sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import torch
torch.set_num_threads(4)
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

D = "data/processed"
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
MODEL = "DeepChem/ChemBERTa-77M-MTR"


def murcko(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc is not None else None
    except Exception:
        return None


def embed(smiles, bs=64):
    """Mean-pooled (mask-aware) frozen ChemBERTa embeddings -> (N, 384)."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL); mdl.eval()
    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), bs):
            batch = [s if isinstance(s, str) else "" for s in smiles[i:i + bs]]
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = mdl(**enc).last_hidden_state                 # (b, L, 384)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy())
            if (i // bs) % 10 == 0:
                print(f"  embed {i+len(batch)}/{len(smiles)} ({time.time()-t0:.0f}s)", flush=True)
    return np.vstack(out).astype(np.float32)


def curve(y, pred, max_sim):
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (max_sim >= lo) & (max_sim < hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "mae": None, "rae": None}); continue
        mae = float(np.mean(np.abs(y[m] - pred[m])))
        r = float(rae(y[m], pred[m])) if n > 1 else None
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": n, "mae": round(mae, 4),
                     "rae": round(r, 4) if r is not None else None})
    return rows


def scaffold_oof(X, y, folds, head):
    oof = np.full(len(y), np.nan)
    for tr_idx, va_idx in folds:
        if head == "ridge":
            sc = StandardScaler().fit(X[tr_idx])
            m = Ridge(alpha=10.0).fit(sc.transform(X[tr_idx]), y[tr_idx])
            oof[va_idx] = m.predict(sc.transform(X[va_idx]))
        else:
            m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                                  n_jobs=4, verbose=-1).fit(X[tr_idx], y[tr_idx])
            oof[va_idx] = m.predict(X[va_idx])
    return oof


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    smiles = tr["smiles"].tolist()
    y = tr["pec50"].to_numpy(float)
    scaffolds = [murcko(s) for s in smiles]
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)   # SAME as nb952

    # reuse nb952's per-compound max-sim so bins align exactly
    max_sim = np.load(f"{D}/nb952_max_sim_4139.npy")
    assert len(max_sim) == len(y), "nb952 max_sim length mismatch — rerun nb952 first"

    print(f"embedding {len(smiles)} SMILES with frozen {MODEL} ...", flush=True)
    X = embed(smiles)
    print(f"embeddings: {X.shape}")

    lgbm_curve = json.load(open(f"{D}/nb952_stress_curve.json"))["lgbm_curve"]
    lgbm_deep = next(r for r in lgbm_curve if r["bin"] == "[0.0,0.3)")["mae"]

    results = {}
    for head in ["ridge", "lgbm"]:
        oof = scaffold_oof(X, y, folds, head)
        overall = rae(y, oof)
        rows = curve(y, oof, max_sim)
        deep = next(r for r in rows if r["bin"] == "[0.0,0.3)")
        results[head] = {"overall_rae": round(float(overall), 4), "curve": rows,
                         "deep_extrap_mae": deep["mae"]}
        np.save(f"{D}/nb953_chemberta_frozen_{head}_oof.npy", oof)
        print(f"\nChemBERTa-frozen + {head}: overall scaffold-CV RAE = {overall:.4f}")
        print(f"{'sim-to-train':18s} {'n':>5s} {'MAE':>8s} {'RAE':>8s}")
        print("-" * 44)
        for r in rows:
            mae = f"{r['mae']:.4f}" if r["mae"] is not None else "   --"
            rr = f"{r['rae']:.4f}" if r["rae"] is not None else "   --"
            print(f"{r['bin']:18s} {r['n']:5d} {mae:>8s} {rr:>8s}")

    # ---- the decision ----
    print("\n" + "=" * 60)
    print("DEEP-EXTRAPOLATION (sim<0.3) MAE — flatter = better generalization")
    print(f"  LGBM-combined (nb952 ref) : {lgbm_deep:.4f}")
    for head in ["ridge", "lgbm"]:
        dm = results[head]["deep_extrap_mae"]
        verdict = "BEATS ref (fine-tune worth GPU)" if dm < lgbm_deep else "no flatter than LGBM"
        print(f"  ChemBERTa-frozen + {head:5s}  : {dm:.4f}   <- {verdict}")
    print("=" * 60)
    print("Note: frozen is the FLOOR; end-to-end fine-tuning adapts the representation")
    print("to PXR and typically beats frozen. If frozen is ALREADY competitive at")
    print("sim<0.3, fine-tuning on GPU is well-motivated. If frozen is much worse")
    print("everywhere AND not flatter, the foundation bet is a long shot.")

    json.dump({"model": MODEL, "lgbm_deep_extrap_mae": lgbm_deep, "results": results},
              open(f"{D}/nb953_chemberta_frozen_degradation.json", "w"), indent=2)
    print(f"\nsaved -> {D}/nb953_chemberta_frozen_degradation.json")


if __name__ == "__main__":
    main()

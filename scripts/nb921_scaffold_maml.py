"""nb921 -- MAML few-shot meta-learning across Murcko scaffold tasks.

Pipeline:
  1) Group 4139 train compounds by Bemis-Murcko scaffold; keep scaffolds
     with >= 8 compounds (one task per scaffold).
  2) Each task: sample 4 support + 4 query examples per meta-iter.
  3) Meta-model: MLP Morgan 2048 -> 256 -> 128 -> 1 (Tanh/ReLU MLP).
  4) MAML (1st-order): inner_lr=0.01, 5 inner steps; outer Adam lr=1e-3.
     200 meta-iters, task batch = 4. Outer loss = mean query MSE after
     inner-adaptation on support.
  5) Test: for each of 513 test compounds find top-1 train scaffold by
     median-Tanimoto-to-scaffold-members, take that scaffold's support
     set (k=4 nearest), do 5 inner steps, predict.
  6) Compute in_RAE on 253 unblind. Save submissions/nb921_scaffold_maml.csv.

Hard wall-time budget: ~12 minutes on CPU. Saves partial state and exits
with success=False if exceeded.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import RDLogger

from pxr.data import load_train, load_test
from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.eval import rae
from pxr.paths import DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

TAG = "nb921"
SEED = 0
N_BITS = 2048
D_H1 = 256
D_H2 = 128
K_SUPPORT = 4
K_QUERY = 4
INNER_LR = 0.01
INNER_STEPS = 5
OUTER_LR = 1e-3
META_ITERS = 200
TASK_BATCH = 4
MIN_TASK_SIZE = 8
WALL_BUDGET_S = 12 * 60  # 12 min hard wall

torch.manual_seed(SEED)
np.random.seed(SEED)

ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")


# ----------------------------------------------------------------------------
# Functional MLP (so we can backprop through manually updated weights for MAML)
# ----------------------------------------------------------------------------
def init_params():
    """Glorot-init MLP params: 2048 -> 256 -> 128 -> 1."""
    g = torch.Generator().manual_seed(SEED)
    def lin(in_d, out_d):
        w = torch.empty(out_d, in_d).uniform_(
            -((6.0 / (in_d + out_d)) ** 0.5),
            ((6.0 / (in_d + out_d)) ** 0.5),
            generator=g,
        )
        b = torch.zeros(out_d)
        return w, b
    w1, b1 = lin(N_BITS, D_H1)
    w2, b2 = lin(D_H1, D_H2)
    w3, b3 = lin(D_H2, 1)
    params = {
        "w1": nn.Parameter(w1), "b1": nn.Parameter(b1),
        "w2": nn.Parameter(w2), "b2": nn.Parameter(b2),
        "w3": nn.Parameter(w3), "b3": nn.Parameter(b3),
    }
    return params


def forward(params, x):
    h = F.relu(F.linear(x, params["w1"], params["b1"]))
    h = F.relu(F.linear(h, params["w2"], params["b2"]))
    y = F.linear(h, params["w3"], params["b3"]).squeeze(-1)
    return y


def inner_adapt(params, x_s, y_s, lr=INNER_LR, steps=INNER_STEPS, create_graph=False):
    """SGD inner loop; returns dict of adapted (fast) weights."""
    fast = {k: v for k, v in params.items()}
    for _ in range(steps):
        yhat = forward(fast, x_s)
        loss = F.mse_loss(yhat, y_s)
        grads = torch.autograd.grad(
            loss, list(fast.values()),
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
        fast = {k: (v - lr * g) for (k, v), g in zip(fast.items(), grads)}
    return fast


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    def elapsed():
        return time.time() - t0
    def out_of_time():
        return elapsed() > WALL_BUDGET_S

    print(f"[{TAG}] start  budget={WALL_BUDGET_S}s")

    # ---- Load data ----
    tr = load_train()
    tr = tr.dropna(subset=["pec50", "smiles"]).reset_index(drop=True)
    # Aggregate replicates per Molecule Name (median pEC50)
    agg = (
        tr.groupby("name", as_index=False)
          .agg(smiles=("smiles", "first"), pec50=("pec50", "median"))
    )
    print(f"train compounds (deduped on name): {len(agg)}")

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    te_smi = te_df["SMILES"].tolist()
    te_names = te_df["Molecule Name"].tolist()
    print(f"test compounds: {len(te_df)}")

    # ---- Scaffolds ----
    print(f"[{TAG}] computing Murcko scaffolds...  t={elapsed():.1f}s")
    agg["scaffold"] = agg["smiles"].map(lambda s: bemis_murcko(s) or "")
    sc_counts = agg["scaffold"].value_counts()
    big_scaffolds = sc_counts[sc_counts >= MIN_TASK_SIZE].index.tolist()
    # Drop empty-string scaffold (failed parses) if present
    big_scaffolds = [s for s in big_scaffolds if s]
    print(f"scaffolds with >= {MIN_TASK_SIZE} compounds: {len(big_scaffolds)}")
    if len(big_scaffolds) < 4:
        print("ERROR: too few scaffolds")
        return {"success": False, "reason": "too_few_scaffolds"}

    # ---- Morgan FPs ----
    print(f"[{TAG}] morgan FPs train...  t={elapsed():.1f}s")
    tr_fp = morgan_fp_batch(agg["smiles"].tolist(), radius=2, n_bits=N_BITS)
    tr_fp_f = tr_fp.astype(np.float32)
    y_arr = agg["pec50"].astype(np.float32).values
    name_to_idx = {n_: i for i, n_ in enumerate(agg["name"])}

    print(f"[{TAG}] morgan FPs test...  t={elapsed():.1f}s")
    te_fp = morgan_fp_batch(te_smi, radius=2, n_bits=N_BITS)
    te_fp_f = te_fp.astype(np.float32)

    # Task index: scaffold -> array of train indices
    sc_to_idx = {}
    for s in big_scaffolds:
        sc_to_idx[s] = agg.index[agg["scaffold"] == s].to_numpy()

    # ---- Init MAML params ----
    params = init_params()
    opt = torch.optim.Adam(params.values(), lr=OUTER_LR)
    rng = np.random.RandomState(SEED)

    # ---- Meta-train ----
    print(f"[{TAG}] meta-train {META_ITERS} iters, task_batch={TASK_BATCH}, "
          f"inner_steps={INNER_STEPS}, inner_lr={INNER_LR}")
    losses = []
    best_state = {k: v.detach().clone() for k, v in params.items()}
    completed_iters = 0
    for it in range(META_ITERS):
        if out_of_time():
            print(f"[{TAG}] WALL EXCEEDED at meta-iter {it}, stopping meta-train")
            break
        opt.zero_grad()
        meta_loss = 0.0
        tasks = rng.choice(len(big_scaffolds), TASK_BATCH, replace=False)
        for ti in tasks:
            sc = big_scaffolds[ti]
            idx_all = sc_to_idx[sc]
            if len(idx_all) < K_SUPPORT + K_QUERY:
                # sample with replacement when scaffold is small
                pick = rng.choice(idx_all, K_SUPPORT + K_QUERY, replace=True)
            else:
                pick = rng.choice(idx_all, K_SUPPORT + K_QUERY, replace=False)
            sup_idx = pick[:K_SUPPORT]
            qry_idx = pick[K_SUPPORT:K_SUPPORT + K_QUERY]
            x_s = torch.from_numpy(tr_fp_f[sup_idx])
            y_s = torch.from_numpy(y_arr[sup_idx])
            x_q = torch.from_numpy(tr_fp_f[qry_idx])
            y_q = torch.from_numpy(y_arr[qry_idx])
            # 1st-order MAML: detach inner-adapted weights (create_graph=False)
            fast = inner_adapt(params, x_s, y_s, create_graph=False)
            yhat_q = forward(fast, x_q)
            q_loss = F.mse_loss(yhat_q, y_q)
            meta_loss = meta_loss + q_loss
        meta_loss = meta_loss / TASK_BATCH
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        opt.step()
        losses.append(float(meta_loss.item()))
        completed_iters = it + 1
        if (it + 1) % 25 == 0 or it == 0:
            recent = np.mean(losses[-25:])
            print(f"  iter {it+1:3d}/{META_ITERS}  meta_loss={recent:.4f}  t={elapsed():.1f}s")
            best_state = {k: v.detach().clone() for k, v in params.items()}

    # ---- Save trained state ----
    torch.save(
        {"state_dict": {k: v.detach() for k, v in params.items()},
         "completed_iters": completed_iters,
         "n_scaffold_tasks": len(big_scaffolds)},
        ART / f"{TAG}_maml.pt",
    )
    print(f"[{TAG}] saved meta-state ({completed_iters} iters)")

    partial = out_of_time()
    if partial:
        print(f"[{TAG}] WALL EXCEEDED -- writing partial outputs, success=False")

    # ---- Predict: for each test, route to top-1 scaffold by mean Tanimoto ----
    # Build per-scaffold OR-bitvector (popcount features), use bit-AND for fast
    # Tanimoto bound to centroid. Faster: per-scaffold mean FP vector and use
    # cosine on bits as the routing similarity.
    print(f"[{TAG}] routing 513 test compounds to scaffold tasks...  t={elapsed():.1f}s")
    sc_centroids = np.zeros((len(big_scaffolds), N_BITS), dtype=np.float32)
    for i, sc in enumerate(big_scaffolds):
        sc_centroids[i] = tr_fp_f[sc_to_idx[sc]].mean(axis=0)
    sc_norm = np.linalg.norm(sc_centroids, axis=1, keepdims=True) + 1e-9
    te_norm = np.linalg.norm(te_fp_f, axis=1, keepdims=True) + 1e-9
    sim = (te_fp_f / te_norm) @ (sc_centroids / sc_norm).T  # (513, T)
    nearest = sim.argmax(axis=1)

    # ---- Inner-adapt per test, then predict ----
    te_pred = np.zeros(len(te_df), dtype=np.float32)
    # Group test indices by task for batched adaptation
    task_to_test = {}
    for ti, tj in enumerate(nearest):
        task_to_test.setdefault(int(tj), []).append(ti)

    for task_id, test_indices in task_to_test.items():
        if out_of_time():
            print(f"[{TAG}] WALL EXCEEDED at predict task {task_id}, stopping")
            partial = True
            break
        sc = big_scaffolds[task_id]
        idx_all = sc_to_idx[sc]
        # Use full scaffold (or all if <= K_SUPPORT) as support set
        if len(idx_all) <= K_SUPPORT:
            sup_idx = idx_all
        else:
            # K_SUPPORT random for stochasticity but seeded
            sup_idx = rng.choice(idx_all, K_SUPPORT, replace=False)
        x_s = torch.from_numpy(tr_fp_f[sup_idx])
        y_s = torch.from_numpy(y_arr[sup_idx])
        # Use current best_state as init
        with torch.enable_grad():
            init_clone = {k: v.detach().clone().requires_grad_(True)
                          for k, v in best_state.items()}
            fast = inner_adapt(init_clone, x_s, y_s,
                               steps=INNER_STEPS, create_graph=False)
        with torch.no_grad():
            x_t = torch.from_numpy(te_fp_f[test_indices])
            yhat = forward({k: v.detach() for k, v in fast.items()}, x_t)
            te_pred[test_indices] = yhat.numpy().astype(np.float32)

    # Clip to train pEC50 range +/- 0.5
    p_lo, p_hi = float(y_arr.min()) - 0.5, float(y_arr.max()) + 0.5
    te_pred = np.clip(te_pred, p_lo, p_hi)

    # ---- Unblind score ----
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    n2i = {n_: i for i, n_ in enumerate(te_names)}
    unb_keep = unb[unb["Molecule Name"].isin(n2i)].reset_index(drop=True)
    unb_idx = np.array([n2i[n_] for n_ in unb_keep["Molecule Name"]], dtype=int)
    unb_y = unb_keep["pEC50"].astype(float).values.astype(np.float64)
    in_rae = float(rae(unb_y, te_pred[unb_idx]))

    print("\n" + "=" * 78)
    print(f"[{TAG}] UNBLIND RAE (n={len(unb_idx)})  = {in_rae:.4f}")
    print(f"  truth mean/std = {unb_y.mean():.3f} / {unb_y.std():.3f}")
    print(f"  pred  mean/std = {te_pred.mean():.3f} / {te_pred.std():.3f}")
    print(f"  scaffold tasks = {len(big_scaffolds)}  meta-iters completed = {completed_iters}")
    print(f"  total time = {elapsed():.1f}s  (budget {WALL_BUDGET_S}s)")
    print("=" * 78)

    # ---- Save ----
    sub = SUBMISSIONS / f"{TAG}_scaffold_maml.csv"
    pd.DataFrame({
        "Molecule Name": te_names,
        "SMILES": te_smi,
        "pEC50": te_pred.astype(np.float32),
    }).to_csv(sub, index=False)
    print(f"Wrote {sub}")

    np.save(ART / f"{TAG}_te_pred.npy", te_pred.astype(np.float32))

    return {
        "success": (not partial),
        "in_rae": in_rae,
        "n_scaffold_tasks": int(len(big_scaffolds)),
        "completed_meta_iters": int(completed_iters),
        "wall_time_s": float(elapsed()),
        "submission": str(sub),
    }


if __name__ == "__main__":
    r = main()
    print("\n==== SUMMARY ====")
    for k, v in r.items():
        print(f"  {k}: {v}")

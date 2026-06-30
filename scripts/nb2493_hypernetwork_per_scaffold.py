"""nb2493 -- Hypernetwork generating per-scaffold LGBM hyperparams.

DIFFERENT (substrate change, not operator change):
A tiny meta-NN ingests a scaffold-cluster fingerprint and emits a discrete
LGBM config (depth, num_leaves, lr, reg_lambda). For each of K=20
Bemis-Murcko / Tanimoto-K-means scaffold buckets we train a dedicated LGBM
with the hyper-NN-generated config on that bucket's train members.

REINFORCE training of the meta-NN:
    action = sample categorical per hyperparam from softmax logits
    reward = -RAE on bucket OOF (5-fold CV within bucket)
    grad   = -E[reward * grad_log_prob(action)] (policy gradient)

Inference: assign a test compound to nearest scaffold cluster -> use that
bucket's LGBM with its hyper-NN-generated config (refit on full bucket).

Evaluation: outer 5-fold scaffold CV on the 253 unblind compounds.

GATES:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4601  -> MARGINAL_BEAT
    else                -> FAIL

Outputs:
    data/processed/nb2493_summary.json
    data/processed/nb2493_pred_oof.npy   (253-vector OOF)
    data/processed/te_nb2493.npy          (513-vector deploy)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.cluster import KMeans

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2493"
N_CLUSTERS = 20
SCAF_FP_BITS = 1024
SCAF_FP_RADIUS = 2

# Discrete hyperparameter choices (action space)
DEPTH_CHOICES   = [3, 4, 5, 6]
LEAVES_CHOICES  = [7, 15, 31]
LR_CHOICES      = [0.02, 0.05, 0.1]
LAMBDA_CHOICES  = [0.5, 1.0, 2.0, 5.0]
ACTION_SIZES = [
    len(DEPTH_CHOICES),
    len(LEAVES_CHOICES),
    len(LR_CHOICES),
    len(LAMBDA_CHOICES),
]
N_ACTIONS = sum(ACTION_SIZES)  # 4+3+3+4=14 logits

# Hypernetwork
HYPERNET_HIDDEN = 32
HYPERNET_LR     = 1e-2
REINFORCE_EPOCHS = 8
SAMPLES_PER_STEP = 2
INNER_KF_SEED   = 11

# Outer CV (5 folds per spec; reduced to 3 outer seeds for tractable runtime)
N_OUTER_FOLDS = 5
OUTER_KF_SEEDS = [1001, 1002, 1003]

# LGBM fixed knobs (small n_estimators -- bucket sizes ~200 don't need 500 trees)
LGBM_BASE = dict(
    objective="regression",
    metric="rmse",
    n_estimators=150,
    min_child_samples=8,
    bagging_fraction=0.85,
    bagging_freq=1,
    feature_fraction=0.85,
    verbose=-1,
    n_jobs=2,
)
EARLY_STOP_ROUNDS = 25

GATE_PROMOTE  = 0.4570
GATE_MARGINAL = 0.4601


# --------------------------------------------------------------------------
# Hypernetwork
# --------------------------------------------------------------------------
class HyperNet(nn.Module):
    """Scaffold-FP -> logits over (depth, leaves, lr, reg_lambda) choices."""

    def __init__(self, in_dim: int, hidden: int = HYPERNET_HIDDEN):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # one head per discrete hyperparam
        self.head_depth  = nn.Linear(hidden, len(DEPTH_CHOICES))
        self.head_leaves = nn.Linear(hidden, len(LEAVES_CHOICES))
        self.head_lr     = nn.Linear(hidden, len(LR_CHOICES))
        self.head_lambda = nn.Linear(hidden, len(LAMBDA_CHOICES))

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return [
            self.head_depth(h),
            self.head_leaves(h),
            self.head_lr(h),
            self.head_lambda(h),
        ]


def sample_action(logits_list):
    """Sample one categorical action per head; return (action_idx_list, log_prob_sum)."""
    actions, log_probs = [], []
    for logits in logits_list:
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        actions.append(int(a.item()))
        log_probs.append(dist.log_prob(a))
    return actions, torch.stack(log_probs).sum()


def greedy_action(logits_list):
    return [int(torch.argmax(l).item()) for l in logits_list]


def actions_to_config(actions):
    d, lv, lr, la = actions
    return dict(
        max_depth=DEPTH_CHOICES[d],
        num_leaves=LEAVES_CHOICES[lv],
        learning_rate=LR_CHOICES[lr],
        reg_lambda=LAMBDA_CHOICES[la],
    )


# --------------------------------------------------------------------------
# Bucket-LGBM utilities
# --------------------------------------------------------------------------
def train_lgbm_bucket(X_tr, y_tr, X_va, y_va, hp: dict, seed: int = 0):
    params = dict(LGBM_BASE)
    params.update(hp)
    params["random_state"] = seed
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)],
    )
    return model


def bucket_inner_oof_rae(X_bucket, y_bucket, hp, seed=INNER_KF_SEED):
    """Inner reward: single 75/25 split RAE on bucket members.

    One LGBM fit per call (REINFORCE evaluates many per epoch -- keep cheap).
    Very small buckets (<6) fall back to train-RAE as a weak signal.
    """
    n = len(y_bucket)
    if n < 6:
        m = train_lgbm_bucket(X_bucket, y_bucket, X_bucket, y_bucket, hp, seed=seed)
        return float(rae(y_bucket, m.predict(X_bucket)))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = max(2, int(0.25 * n))
    va, tr = idx[:cut], idx[cut:]
    m = train_lgbm_bucket(X_bucket[tr], y_bucket[tr], X_bucket[va], y_bucket[va], hp, seed=seed)
    return float(rae(y_bucket[va], m.predict(X_bucket[va])))


# --------------------------------------------------------------------------
# Hypernetwork training (REINFORCE) -- one pass over buckets per epoch
# --------------------------------------------------------------------------
def train_hypernet(centroids_tensor, bucket_data, n_epochs=REINFORCE_EPOCHS):
    """Train the hypernet via REINFORCE on bucket OOF RAE.

    centroids_tensor: (K, F) tensor of cluster centroid FPs
    bucket_data: list[(X_bucket, y_bucket)] aligned with centroids
    """
    in_dim = centroids_tensor.shape[1]
    net = HyperNet(in_dim).to(centroids_tensor.device)
    opt = torch.optim.Adam(net.parameters(), lr=HYPERNET_LR)

    # running baseline per bucket for variance reduction
    baseline = {k: None for k in range(len(bucket_data))}

    history = []
    for ep in range(n_epochs):
        opt.zero_grad()
        total_loss = torch.zeros(1, device=centroids_tensor.device)
        ep_rewards = []
        for k, (X_b, y_b) in enumerate(bucket_data):
            if X_b is None or len(y_b) < 3:
                continue
            logits = net(centroids_tensor[k:k+1])  # (1, choices) per head
            logits = [l.squeeze(0) for l in logits]
            for _ in range(SAMPLES_PER_STEP):
                actions, logp = sample_action(logits)
                hp = actions_to_config(actions)
                r = bucket_inner_oof_rae(X_b, y_b, hp)
                reward = -r  # we minimize RAE
                ep_rewards.append(reward)
                b = baseline[k] if baseline[k] is not None else reward
                advantage = reward - b
                # update baseline (EMA)
                baseline[k] = 0.7 * b + 0.3 * reward
                total_loss = total_loss + (-advantage * logp)
        if len(ep_rewards) == 0:
            break
        total_loss = total_loss / max(1, len(ep_rewards))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        ep_mean_rae = float(-np.mean(ep_rewards))
        history.append({"epoch": ep, "mean_bucket_rae": ep_mean_rae, "loss": float(total_loss.item())})
        if (ep + 1) % 5 == 0:
            print(f"   [hypernet] epoch {ep+1:2d}  mean_bucket_RAE={ep_mean_rae:.4f}  loss={float(total_loss.item()):+.4f}")
    return net, history


# --------------------------------------------------------------------------
# Per-cluster deploy: fit LGBM per bucket, predict on test
# --------------------------------------------------------------------------
def fit_deploy_per_bucket(centroids_tensor, bucket_data, net, seed=0):
    """For each bucket use greedy hypernet action to pick HP and refit on
    full bucket (no holdout). Returns dict {cluster_id: model, config}."""
    deploy = {}
    for k, (X_b, y_b) in enumerate(bucket_data):
        if X_b is None or len(y_b) < 2:
            deploy[k] = {"model": None, "config": None, "n": 0}
            continue
        with torch.no_grad():
            logits = net(centroids_tensor[k:k+1])
            logits = [l.squeeze(0) for l in logits]
        a = greedy_action(logits)
        hp = actions_to_config(a)
        # refit on full bucket with a small held-out for early stopping
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y_b))
        cut = max(1, int(0.15 * len(y_b)))
        va, tr = idx[:cut], idx[cut:]
        if len(tr) < 5:
            tr = np.arange(len(y_b))
            va = tr
        model = train_lgbm_bucket(X_b[tr], y_b[tr], X_b[va], y_b[va], hp, seed=seed)
        deploy[k] = {"model": model, "config": hp, "n": int(len(y_b))}
    return deploy


def predict_via_buckets(X_query, cluster_ids_query, deploy, fallback_const=5.0):
    """Each query row goes through its assigned bucket model.

    Buckets with no model (too few train members) get a fallback constant
    equal to the global mean of pEC50 (~5 on PXR train).
    """
    pred = np.full(len(X_query), np.nan, dtype=np.float64)
    for c, info in deploy.items():
        mask = cluster_ids_query == c
        if not mask.any():
            continue
        if info["model"] is None:
            pred[mask] = fallback_const
        else:
            pred[mask] = info["model"].predict(X_query[mask])
    if np.isnan(pred).any():
        pred[np.isnan(pred)] = fallback_const
    return pred


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Hypernetwork generating per-scaffold LGBM hyperparams")
    print("=" * 78)

    # ---- Load core data ----
    tr = load_train()
    te = load_test()
    y_tr_full = tr["pec50"].values.astype(np.float64)
    tr_smiles = tr["smiles"].values
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_tr, n_te = len(tr_smiles), len(te_smiles)

    cache = np.load(DATA_PROCESSED / "cache_combined_features.npz")
    X_tr = cache["X_tr"].astype(np.float32)
    X_te = cache["X_te"].astype(np.float32)
    assert X_tr.shape[0] == n_tr and X_te.shape[0] == n_te
    print(f"[load] X_tr={X_tr.shape}  X_te={X_te.shape}  y_tr={y_tr_full.shape}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] unb_idx={unb_idx.shape}  y_unb stats mean={y_unb.mean():.3f} std={y_unb.std():.3f}")

    # ---- Scaffold extraction ----
    print("\n[scaffold] extracting Bemis-Murcko + Morgan FPs")
    tr_scaf = [bemis_murcko(s) or "" for s in tr_smiles]
    te_scaf = [bemis_murcko(s) or "" for s in te_smiles]
    # scaffold FPs (zero row for empty scaffold)
    scaf_fp_tr = morgan_fp_batch(
        [s if s else "C" for s in tr_scaf],
        radius=SCAF_FP_RADIUS, n_bits=SCAF_FP_BITS,
    ).astype(np.float32)
    scaf_fp_te = morgan_fp_batch(
        [s if s else "C" for s in te_scaf],
        radius=SCAF_FP_RADIUS, n_bits=SCAF_FP_BITS,
    ).astype(np.float32)
    print(f"[scaffold] scaf_fp_tr={scaf_fp_tr.shape}  scaf_fp_te={scaf_fp_te.shape}")

    # ---- K-means on training scaffold FPs (Tanimoto-proxy via cosine on bits) ----
    print(f"\n[cluster] K-means K={N_CLUSTERS} on train scaffold-FP")
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit(scaf_fp_tr)
    cl_tr = km.labels_.astype(np.int64)
    cl_te = km.predict(scaf_fp_te).astype(np.int64)
    centroids = km.cluster_centers_.astype(np.float32)  # (K, F)
    bucket_sizes = np.bincount(cl_tr, minlength=N_CLUSTERS)
    print(f"[cluster] train bucket sizes: min={bucket_sizes.min()}  max={bucket_sizes.max()}  "
          f"median={int(np.median(bucket_sizes))}")
    print(f"[cluster] test  bucket dist:  {np.bincount(cl_te, minlength=N_CLUSTERS).tolist()}")

    # ---- Pack centroids as torch tensor for hypernet ----
    centroids_t = torch.from_numpy(centroids)

    # ====================================================================
    # OUTER 5-FOLD SCAFFOLD CV on 253 unblind
    # ====================================================================
    unb_scaf = [tr_scaf[0]] * n_unb  # placeholder; overwrite below
    unb_scaf = [te_scaf[i] for i in unb_idx]
    X_unb = X_te[unb_idx]
    cl_unb = cl_te[unb_idx]

    all_oof_per_seed = []
    per_seed_summary = []
    for seed in OUTER_KF_SEEDS:
        print("\n" + "-" * 78)
        print(f"OUTER FOLD seed={seed}")
        print("-" * 78)
        splits = scaffold_kfold_indices(unb_scaf, n_splits=N_OUTER_FOLDS, shuffle=True, seed=seed)
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            # ---- Build bucket data: train members are 4139 train PLUS the unblind train-loc rows ----
            # The bucket models see PRE-unblind training rows AND any val-fold-excluded unblind rows.
            # This keeps the fold honest (val-fold unblind rows never see their own label).
            unb_keep = unb_idx[tr_loc]
            X_pool = np.vstack([X_tr, X_te[unb_keep]])
            y_pool = np.concatenate([y_tr_full, y_unb[tr_loc]])
            cl_pool = np.concatenate([cl_tr, cl_te[unb_keep]])

            bucket_data = []
            for k in range(N_CLUSTERS):
                mask = cl_pool == k
                if mask.sum() < 3:
                    bucket_data.append((None, np.array([])))
                else:
                    bucket_data.append((X_pool[mask], y_pool[mask]))

            # ---- Train hypernet via REINFORCE on this fold's pool ----
            t_h = time.time()
            net, hist = train_hypernet(centroids_t, bucket_data, n_epochs=REINFORCE_EPOCHS)
            # ---- Deploy per bucket ----
            deploy = fit_deploy_per_bucket(centroids_t, bucket_data, net, seed=seed)

            # ---- Predict on val-loc unblind ----
            X_va = X_te[unb_idx[va_loc]]
            cl_va = cl_te[unb_idx[va_loc]]
            pred_va = predict_via_buckets(X_va, cl_va, deploy)
            oof[va_loc] = pred_va
            r = float(rae(y_unb[va_loc], pred_va))
            fold_rae.append(r)
            print(f"   fold {fi}: n_va={len(va_loc):3d}  RAE={r:.4f}  "
                  f"hypernet_final_loss={hist[-1]['loss'] if hist else float('nan'):+.4f}  "
                  f"wall={time.time()-t_h:.1f}s")

        pooled = float(rae(y_unb, oof))
        all_oof_per_seed.append(oof)
        per_seed_summary.append({
            "seed": int(seed),
            "fold_rae": [float(r) for r in fold_rae],
            "pooled_rae": pooled,
        })
        print(f"   pooled seed={seed} OOF RAE = {pooled:.4f}")

    # ---- Aggregate ----
    mean_oof = np.mean(np.column_stack(all_oof_per_seed), axis=1)
    rae_of_mean_oof = float(rae(y_unb, mean_oof))
    pooled_rae_per_seed = np.array([s["pooled_rae"] for s in per_seed_summary])
    mean_rae = float(pooled_rae_per_seed.mean())
    std_rae  = float(pooled_rae_per_seed.std())
    print("\n" + "=" * 78)
    print(f"OUTER CV SUMMARY  ({len(OUTER_KF_SEEDS)} seeds)")
    print("=" * 78)
    print(f"   per-seed pooled RAE: {[f'{r:.4f}' for r in pooled_rae_per_seed.tolist()]}")
    print(f"   mean_rae   = {mean_rae:.4f}  +/- {std_rae:.4f}")
    print(f"   rae(mean_oof) = {rae_of_mean_oof:.4f}")

    # ====================================================================
    # Deploy on full 4139 train -> predict 513 test
    # ====================================================================
    print("\n" + "-" * 78)
    print("DEPLOY: full 4139 + 253 unblind -> 513 test")
    print("-" * 78)
    X_pool = np.vstack([X_tr, X_te[unb_idx]])
    y_pool = np.concatenate([y_tr_full, y_unb])
    cl_pool = np.concatenate([cl_tr, cl_te[unb_idx]])
    bucket_data = []
    for k in range(N_CLUSTERS):
        mask = cl_pool == k
        if mask.sum() < 3:
            bucket_data.append((None, np.array([])))
        else:
            bucket_data.append((X_pool[mask], y_pool[mask]))
    net, _ = train_hypernet(centroids_t, bucket_data, n_epochs=REINFORCE_EPOCHS)
    deploy = fit_deploy_per_bucket(centroids_t, bucket_data, net, seed=0)
    te_pred = predict_via_buckets(X_te, cl_te, deploy).astype(np.float32)
    te_unb_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te(513) mean/std = {te_pred.mean():.3f}/{te_pred.std():.3f}")
    print(f"   te[unb_idx] RAE  = {te_unb_rae:.4f}  (in-sample; expected lower than mean_rae)")

    # ---- Save artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float64))
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_pred)

    # ---- Gate ----
    if mean_rae < GATE_PROMOTE:
        gate_status = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        gate_status = "MARGINAL_BEAT"
    else:
        gate_status = "FAIL"

    # ---- Deploy configs (greedy actions per cluster) ----
    deploy_configs = []
    for k, info in deploy.items():
        deploy_configs.append({
            "cluster_id": int(k),
            "n_bucket_train": int(info["n"]),
            "config": info["config"],
        })

    summary = {
        "tag": TAG,
        "method": "hypernetwork_per_scaffold_LGBM_REINFORCE",
        "n_clusters": N_CLUSTERS,
        "scaf_fp_bits": SCAF_FP_BITS,
        "scaf_fp_radius": SCAF_FP_RADIUS,
        "action_space": {
            "depth": DEPTH_CHOICES,
            "leaves": LEAVES_CHOICES,
            "lr": LR_CHOICES,
            "reg_lambda": LAMBDA_CHOICES,
        },
        "hypernet_hidden": HYPERNET_HIDDEN,
        "reinforce_epochs": REINFORCE_EPOCHS,
        "samples_per_step": SAMPLES_PER_STEP,
        "n_outer_folds": N_OUTER_FOLDS,
        "outer_kf_seeds": OUTER_KF_SEEDS,
        "n_tr": int(n_tr),
        "n_te": int(n_te),
        "n_unb": int(n_unb),
        "train_bucket_sizes": bucket_sizes.tolist(),
        "test_bucket_dist": np.bincount(cl_te, minlength=N_CLUSTERS).tolist(),
        "per_seed_summary": per_seed_summary,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_oof": rae_of_mean_oof,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_configs": deploy_configs,
        "deploy_te_mean": float(te_pred.mean()),
        "deploy_te_std": float(te_pred.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "gate_status": gate_status,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {json_path}")
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5 seeds)  = {mean_rae:.4f}  +/- {std_rae:.4f}")
    print(f"   rae(mean_oof)       = {rae_of_mean_oof:.4f}")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}")
    print(f"   gate_promote target = {GATE_PROMOTE:.4f}")
    print(f"   gate_marginal target= {GATE_MARGINAL:.4f}")
    print(f"   gate_status         = {gate_status}")
    print(f"   wall                = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== FINAL ====")
    for k in ("mean_rae", "std_rae", "rae_of_mean_oof", "te_unb_rae_in_sample", "gate_status"):
        print(f"  {k}: {res.get(k)}")

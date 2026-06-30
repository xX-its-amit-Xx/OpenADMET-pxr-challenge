"""nb2472 -- Genetic Algorithm search over blend weights.

Different optimizer than SLSQP -- GA explores the simplex via crossover and
mutation instead of gradient-following local search. The hypothesis is that
the SLSQP simplex landscape on n=253 with K=5 anchors has many near-optimal
basins (SLSQP collapses to <=2 anchors); GA may discover off-corner solutions
that beat SLSQP under scaffold 5-fold CV.

ANCHOR POOL (K=5):
    0. nb2240_K20      data/processed/nb2240_mean_bag_oof_K20.npy
    1. nb1191          RECONSTRUCT (no raw nb1191_oof.npy on disk)
    2. nb730           data/processed/nb730_pred_oof.npy
    3. nb562           data/processed/nb562_pred_oof.npy
    4. chemprop_aux    data/processed/nb1133_chemprop_aux_pred_oof.npy

PROTOCOL:
    5-fold scaffold CV; for each fold:
      - GA: population=50, generations=100, tournament k=3, cxpb=0.7, mutpb=0.2
      - chromosome = K-element simplex (cxBlend + simplex-re-project mutation)
      - fitness = -RAE on fold-train OOF
    Compare against SLSQP simplex baseline on same anchor pool.

GATE: mean_rae < 0.4570 -> PROMOTE, else FAIL.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

try:
    from deap import base, creator, tools
except Exception as e:
    print(f"[install] deap import failed: {e}")
    summary = {"tag": "nb2472", "status": "INSTALL_FAILED", "error": str(e)}
    (DATA_PROCESSED / "nb2472_summary.json").write_text(json.dumps(summary, indent=2))
    sys.exit(0)

TAG = "nb2472"
GATE_RAE = 0.4570
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# GA hyperparams
POP_SIZE = 50
N_GEN = 100
TOURN_K = 3
CXPB = 0.7
MUTPB = 0.2
MUT_SIGMA = 0.10  # Gaussian mutation scale per gene

# nb1191 reconstruction params (mirrors nb2171)
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]

# Anchor table: (display_name, oof_path_relative or _RECONSTRUCT, te_path_relative)
ANCHORS = [
    ("nb2240_K20",   "nb2240_mean_bag_oof_K20.npy",       "te_nb2240_K20.npy"),
    ("nb1191",       "_RECONSTRUCT_nb1191_oof",           "te_nb1191.npy"),
    ("nb730",        "nb730_pred_oof.npy",                "te_nb730.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
]


# --------------------------------------------------------------------------
# Reconstruction helpers
# --------------------------------------------------------------------------
def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,)
        cols.append(v)
    return np.column_stack(cols) @ np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)


def reconstruct_nb1191_oof(n_unb: int) -> np.ndarray:
    chemprop_oof = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"] * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"] * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"] * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


# --------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------
def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def _project_simplex(w):
    """Project vector w onto the K-1 simplex (non-negative, sum=1)."""
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.full(len(w), 1.0 / len(w))
    return w / s


def _setup_deap(K: int):
    """Build DEAP toolbox once; clear creator namespace if it exists."""
    # Clear any previously registered classes from another script run
    for name in ("FitnessMin", "Individual"):
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    def init_individual():
        # Dirichlet(1,...,1) = uniform on simplex
        w = np.random.dirichlet(np.ones(K))
        return creator.Individual(w.tolist())

    def mate(ind1, ind2):
        # cxBlend then re-project both children to simplex
        tools.cxBlend(ind1, ind2, alpha=0.3)
        ind1[:] = _project_simplex(ind1).tolist()
        ind2[:] = _project_simplex(ind2).tolist()
        return ind1, ind2

    def mutate(ind):
        # Gaussian per-gene then re-project to simplex
        for i in range(len(ind)):
            ind[i] = ind[i] + random.gauss(0.0, MUT_SIGMA)
        ind[:] = _project_simplex(ind).tolist()
        return (ind,)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=TOURN_K)
    return toolbox


def ga_simplex(P: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Run GA on the simplex, return the best chromosome's weights."""
    K = P.shape[1]
    random.seed(seed)
    np.random.seed(seed)
    toolbox = _setup_deap(K)

    def evaluate(ind):
        w = _project_simplex(ind)
        return (float(np.sum((P @ w - y) ** 2)),)

    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    for _gen in range(N_GEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = [creator.Individual(list(x)) for x in offspring]

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                if c1.fitness.valid:
                    del c1.fitness.values
                if c2.fitness.valid:
                    del c2.fitness.values

        # Mutation
        for m in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(m)
                if m.fitness.valid:
                    del m.fitness.values

        # Evaluate the invalid ones
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Elitism: carry best from old population
        old_best = tools.selBest(pop, 1)[0]
        pop[:] = offspring
        worst_idx = max(range(len(pop)), key=lambda i: pop[i].fitness.values[0])
        if old_best.fitness.values[0] < pop[worst_idx].fitness.values[0]:
            pop[worst_idx] = creator.Individual(list(old_best))
            pop[worst_idx].fitness.values = old_best.fitness.values

    best = tools.selBest(pop, 1)[0]
    return _project_simplex(best)


# --------------------------------------------------------------------------
# CV runners
# --------------------------------------------------------------------------
def cv_run(P_unb, y_unb, unb_scaffolds, kf_seed, optimizer_fn):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof = np.full(n_unb, np.nan)
    fold_w = []
    for fold_idx, (tr_loc, va_loc) in enumerate(splits):
        # Stable per-fold seed for GA reproducibility
        ga_seed = kf_seed * 100 + fold_idx
        w_f = optimizer_fn(P_unb[tr_loc], y_unb[tr_loc], ga_seed) \
            if optimizer_fn is ga_simplex \
            else optimizer_fn(P_unb[tr_loc], y_unb[tr_loc])
        oof[va_loc] = P_unb[va_loc] @ w_f
        fold_w.append(w_f)
    pooled = float(rae(y_unb, oof))
    return pooled, oof, fold_w


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Genetic Algorithm blend search (DEAP)")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    print(f"[load] n_te={n_te}  n_unb={n_unb}  uniq_scaf={len({s for s in unb_scaffolds if s})}")

    # ---- Anchors ----
    oof_cols, te_cols, indiv_rae = [], [], {}
    print("\n[anchors]")
    for disp, oof_rel, te_rel in ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1191_oof":
            oof = reconstruct_nb1191_oof(n_unb)
        else:
            oof_p = DATA_PROCESSED / oof_rel
            assert oof_p.exists(), f"missing OOF: {oof_p}"
            oof = np.load(oof_p).astype(np.float64)
        te_p = DATA_PROCESSED / te_rel
        assert te_p.exists(), f"missing te: {te_p}"
        te_arr = np.load(te_p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_te,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  K={K}")

    # =================================================================
    # SLSQP baseline
    # =================================================================
    print("\n" + "-" * 78)
    print("BASELINE: SLSQP simplex (scaffold 5-fold CV)")
    print("-" * 78)
    slsqp_per_seed = []
    slsqp_oofs = []
    for seed in KF_SEEDS:
        pooled, oof, fw = cv_run(P_unb, y_unb, unb_scaffolds, seed, slsqp_simplex)
        slsqp_per_seed.append({"seed": int(seed), "rae": float(pooled),
                               "w_mean": [float(x) for x in np.mean(fw, axis=0)]})
        slsqp_oofs.append(oof)
        print(f"   seed={seed}  pooled_RAE={pooled:.4f}  w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")
    slsqp_mean_rae = float(np.mean([r["rae"] for r in slsqp_per_seed]))
    slsqp_std_rae = float(np.std([r["rae"] for r in slsqp_per_seed]))
    print(f"\n[slsqp] mean RAE = {slsqp_mean_rae:.4f} (+/- {slsqp_std_rae:.4f})")

    # =================================================================
    # GA optimizer
    # =================================================================
    print("\n" + "-" * 78)
    print(f"GA: pop={POP_SIZE}  gen={N_GEN}  tourn={TOURN_K}  cxpb={CXPB}  mutpb={MUTPB}")
    print("-" * 78)
    ga_per_seed = []
    ga_oofs = []
    for seed in KF_SEEDS:
        pooled, oof, fw = cv_run(P_unb, y_unb, unb_scaffolds, seed, ga_simplex)
        ga_per_seed.append({"seed": int(seed), "rae": float(pooled),
                            "w_mean": [float(x) for x in np.mean(fw, axis=0)]})
        ga_oofs.append(oof)
        print(f"   seed={seed}  pooled_RAE={pooled:.4f}  w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")
    ga_mean_rae = float(np.mean([r["rae"] for r in ga_per_seed]))
    ga_std_rae = float(np.std([r["rae"] for r in ga_per_seed]))
    print(f"\n[ga] mean RAE = {ga_mean_rae:.4f} (+/- {ga_std_rae:.4f})")

    # =================================================================
    # Deploy: refit GA weights on the full unblind 253
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY: GA refit on full 253")
    print("-" * 78)
    w_deploy = ga_simplex(P_unb, y_unb, seed=42)
    blend_unb = P_unb @ w_deploy
    in_rae = float(rae(y_unb, blend_unb))
    deploy_te = (P_te @ w_deploy).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    w_str = ", ".join(f"{disp}={w:.4f}" for (disp, _, _), w in zip(ANCHORS, w_deploy))
    print(f"   deploy weights = {w_str}")
    print(f"   in-sample RAE  = {in_rae:.4f}")
    print(f"   te[unb] RAE    = {te_unb_rae:.4f}")
    print(f"   te(513) m/s    = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    mean_oof_ga = np.mean(np.column_stack(ga_oofs), axis=1)
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, mean_oof_ga.astype(np.float32))
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {pred_oof_path}")
    print(f"[save] {te_npy_path}")

    # =================================================================
    # Gate
    # =================================================================
    gate_pass = ga_mean_rae < GATE_RAE
    status = "PROMOTE" if gate_pass else "FAIL"
    delta_vs_slsqp = ga_mean_rae - slsqp_mean_rae

    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   GA mean RAE  = {ga_mean_rae:.4f}  (target < {GATE_RAE:.4f})")
    print(f"   SLSQP mean   = {slsqp_mean_rae:.4f}  (delta GA - SLSQP = {delta_vs_slsqp:+.4f})")
    print(f"   status       = {status}")

    summary = {
        "tag": TAG,
        "status": status,
        "method": "ga_deap_pop50_gen100_tourn3_cxpb0.7_mutpb0.2_cxBlend_simplex_reproject",
        "ga_hyperparams": {
            "population": POP_SIZE,
            "generations": N_GEN,
            "tournament_size": TOURN_K,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "mut_sigma": MUT_SIGMA,
            "crossover": "cxBlend(alpha=0.3)",
            "mutation": "gaussian_per_gene_then_simplex_project",
            "selection": "tournament",
        },
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "indiv_oof_rae_unb": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_te,
        "slsqp_per_seed": slsqp_per_seed,
        "slsqp_mean_rae": slsqp_mean_rae,
        "slsqp_std_rae": slsqp_std_rae,
        "ga_per_seed": ga_per_seed,
        "ga_mean_rae": ga_mean_rae,
        "ga_std_rae": ga_std_rae,
        "delta_ga_minus_slsqp": delta_vs_slsqp,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(ANCHORS, w_deploy)
        ],
        "in_sample_rae_overfit_bound": in_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path),
        "pred_oof_path": str(pred_oof_path),
        "gate_rae_target": GATE_RAE,
        "gate_pass": bool(gate_pass),
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   GA mean RAE      = {ga_mean_rae:.4f} +/- {ga_std_rae:.4f}")
    print(f"   SLSQP mean RAE   = {slsqp_mean_rae:.4f} +/- {slsqp_std_rae:.4f}")
    print(f"   delta GA-SLSQP   = {delta_vs_slsqp:+.4f}")
    print(f"   gate (<{GATE_RAE})  = {status}")
    print(f"   wall             = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    main()

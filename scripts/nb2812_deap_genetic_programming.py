"""nb2812 -- DEAP symbolic regression (genetic programming) on chemprop-anchor residual.

DIFFERENT PARADIGM from nb2461 (PySR/Julia) and nb2472 (GA blend weights):
    nb2461: PySR symbolic regression via Julia backend; deterministic-search GP.
    nb2472: DEAP GA on simplex blend weights (5 fixed anchors).
    nb2812: DEAP gp.PrimitiveSet symbolic regression in PURE PYTHON, evolving
            arithmetic expression trees over 9 physchem features. No Julia, no
            anchor blend -- a fresh evolved closed-form function for the
            chemprop_aux residual.

ANCHOR: nb2240_K20 mean-bag (28-feat SHAP LGBM on chemprop_aux), the
        established cycle-160+ K=20 anchor used by nb2461.

TARGET: y_unb - oof_anchor (the chemprop-anchor residual on n=253 unblind).

FEATURES (9 physchem):
    MW, LogP, TPSA, HBD, HBA, RotBonds, fsp3, n_rings, n_aromatic_rings

DEAP PROTOCOL:
    PrimitiveSet with binary {add, sub, mul, protected_div} and unary
    {sin, cos, protected_exp, protected_log, protected_sqrt}. Tree-based GP via
    DEAP `gp` module. Population=300, generations=50, tournament k=3,
    cxOnePoint (subtree crossover), mutation = mutShrink + mutUniform (random
    subtree). Fitness = fold-train MSE on residual (RAE-equivalent for blend).

5-fold scaffold CV on the 253 unblind set with kf_seed=1001.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4598 -> MARGINAL_BEAT
    else            -> FAIL

Outputs:
    data/processed/nb2812_summary.json
    data/processed/nb2812_pred_oof.npy   (n=253)
    data/processed/te_nb2812.npy         (n=513)
    best evolved expression per fold + deploy expression in summary.
"""
from __future__ import annotations

import json
import operator
import os
import random
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"
SUMMARY_PATH = PROC / "nb2812_summary.json"

sys.path.insert(0, str(REPO / "src"))


TAG = "nb2812"
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
N_FOLDS = 5
KF_SEED = 1001

# GP hyperparams (per task spec)
POP_SIZE = 300
N_GEN = 50
TOURN_K = 3
CXPB = 0.7      # crossover prob (DEAP-canonical default)
MUTPB = 0.2     # mutation prob
INIT_MIN_DEPTH = 1
INIT_MAX_DEPTH = 3
MAX_TREE_HEIGHT = 12   # static-limit on bloat


def save_summary(status: str, payload: dict | None = None) -> None:
    out = {
        "tag": TAG,
        "status": status,
        "method": "deap_gp_symbolic_regression",
        "notebook": TAG,
    }
    if payload:
        out.update(payload)
    SUMMARY_PATH.write_text(json.dumps(out, indent=2, default=str))


# ---------- import DEAP gp ----------
try:
    from deap import algorithms, base, creator, gp, tools
except Exception as e:
    save_summary("INSTALL_FAILED", {"error": f"deap import failed: {e}"})
    print(f"[install] deap import failed: {e}")
    sys.exit(0)


# ---------- substrate load ----------
te_anchor = np.load(PROC / "te_nb2240_K20.npy")              # (513,)
oof_anchor = np.load(PROC / "nb2240_mean_bag_oof_K20.npy")   # (253,)
y_unb = np.load(PROC / "_audit_unblind_y.npy")               # (253,)
unb_idx = np.load(PROC / "_audit_unblind_idx.npy")           # (253,) into 0..512

assert te_anchor.shape == (513,), f"te_anchor shape {te_anchor.shape}"
assert oof_anchor.shape == y_unb.shape == unb_idx.shape == (253,)

resid_unb = (y_unb - oof_anchor).astype(np.float64)


# ---------- physchem features (9 cols) ----------
from pxr.chem import compute_physchem, bemis_murcko  # noqa: E402

test_df = pd.read_csv(RAW / "pxr-challenge_TEST_BLINDED.csv")
all_te_smiles = test_df["SMILES"].tolist()
unb_smiles = [all_te_smiles[i] for i in unb_idx]

PHYS_KEYS = ["mw", "logp", "tpsa", "hbd", "hba", "rotbonds", "fsp3",
             "n_rings", "n_aromatic_rings"]
N_FEAT = len(PHYS_KEYS)


def phys_row(smi):
    p = compute_physchem(smi)
    if p is None:
        return [np.nan] * N_FEAT
    return [p.get(k, np.nan) for k in PHYS_KEYS]


def _build_X(smiles_list, fill_med=None):
    X = np.array([phys_row(s) for s in smiles_list], dtype=np.float64)
    if fill_med is None:
        fill_med = np.nanmedian(X, axis=0)
    ix_nan = np.isnan(X)
    if ix_nan.any():
        rows, cols = np.where(ix_nan)
        X[rows, cols] = fill_med[cols]
    return X, fill_med


X_unb, col_med = _build_X(unb_smiles, fill_med=None)
X_te, _ = _build_X(all_te_smiles, fill_med=col_med)
assert X_unb.shape == (253, N_FEAT)
assert X_te.shape == (513, N_FEAT)


# ---------- scaffolds for CV ----------
unb_scaffolds = [bemis_murcko(s) or f"__sing_{i}__" for i, s in enumerate(unb_smiles)]
from pxr.eval import rae, scaffold_kfold_indices  # noqa: E402
folds = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED)
print(f"[load] n_unb=253  n_te=513  K_feat={N_FEAT}  folds={N_FOLDS}  kf_seed={KF_SEED}")


# ----------------------------------------------------------------------
# Protected operators (DEAP idiom: avoid div-by-zero / log-of-neg blow-ups)
# ----------------------------------------------------------------------
def protectedDiv(a, b):
    if abs(b) < 1e-9:
        return 1.0
    try:
        v = a / b
        if not np.isfinite(v) or abs(v) > 1e6:
            return 1.0
        return v
    except Exception:
        return 1.0


def protectedLog(a):
    try:
        if a <= 1e-9:
            return 0.0
        v = np.log(a)
        if not np.isfinite(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def protectedExp(a):
    try:
        if a > 50.0:
            return float(np.exp(50.0))
        if a < -50.0:
            return float(np.exp(-50.0))
        v = np.exp(a)
        if not np.isfinite(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def protectedSqrt(a):
    try:
        v = np.sqrt(abs(a))
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def protectedSin(a):
    try:
        return float(np.sin(a))
    except Exception:
        return 0.0


def protectedCos(a):
    try:
        return float(np.cos(a))
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# DEAP PrimitiveSet setup
# ----------------------------------------------------------------------
def _setup_gp():
    """Build the DEAP PrimitiveSet and toolbox. Idempotent across runs."""
    pset = gp.PrimitiveSet("MAIN", N_FEAT)
    # Binary
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2, name="pdiv")
    # Unary
    pset.addPrimitive(protectedSin, 1, name="sin")
    pset.addPrimitive(protectedCos, 1, name="cos")
    pset.addPrimitive(protectedExp, 1, name="pexp")
    pset.addPrimitive(protectedLog, 1, name="plog")
    pset.addPrimitive(protectedSqrt, 1, name="psqrt")
    # Ephemeral constant
    pset.addEphemeralConstant("rand_const", lambda: random.uniform(-1.0, 1.0))
    # Rename variables for readability in evolved expressions
    rename_map = {f"ARG{i}": PHYS_KEYS[i] for i in range(N_FEAT)}
    pset.renameArguments(**rename_map)

    # Clear previously registered classes (script may be re-imported in same proc)
    for name in ("FitnessMin", "Individual"):
        if hasattr(creator, name):
            delattr(creator, name)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
                   pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=INIT_MIN_DEPTH, max_=INIT_MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Crossover: subtree crossover (cxOnePoint operates on PrimitiveTree)
    toolbox.register("mate", gp.cxOnePoint)
    # Mutation operators - we pick stochastically (shrink OR uniform random subtree)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mut_shrink", gp.mutShrink)
    toolbox.register("mut_uniform", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=TOURN_K)

    # Bloat control
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),
                                            max_value=MAX_TREE_HEIGHT))
    toolbox.decorate("mut_uniform", gp.staticLimit(key=operator.attrgetter("height"),
                                                    max_value=MAX_TREE_HEIGHT))
    toolbox.decorate("mut_shrink", gp.staticLimit(key=operator.attrgetter("height"),
                                                   max_value=MAX_TREE_HEIGHT))
    return pset, toolbox


def _predict_tree(individual, toolbox, X):
    """Evaluate a compiled tree on every row of X. Returns float64 (n,) array."""
    func = toolbox.compile(expr=individual)
    out = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        try:
            v = float(func(*X[i].tolist()))
            if not np.isfinite(v):
                v = 0.0
            out[i] = v
        except Exception:
            out[i] = 0.0
    return out


def _make_eval(X_tr, y_tr, toolbox):
    """Return DEAP fitness function: SSE on training rows (lower is better)."""
    def evalSym(individual):
        yhat = _predict_tree(individual, toolbox, X_tr)
        # MSE
        err = float(np.mean((yhat - y_tr) ** 2))
        if not np.isfinite(err):
            err = 1e9
        return (err,)
    return evalSym


def evolve_one_fold(X_tr, y_tr, fold_seed: int):
    """Run one GP evolutionary loop. Return (best_individual, toolbox)."""
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    pset, toolbox = _setup_gp()
    toolbox.register("evaluate", _make_eval(X_tr, y_tr, toolbox))

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    # eaSimple is the canonical DEAP loop (generational, tournament select,
    # then mate / mutate). We pick the mutation operator stochastically by
    # registering a wrapper that randomly chooses shrink vs uniform.
    def mut_combo(individual):
        if random.random() < 0.5:
            return toolbox.mut_shrink(individual)
        return toolbox.mut_uniform(individual)

    toolbox.register("mutate", mut_combo)

    pop, _log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=CXPB, mutpb=MUTPB,
        ngen=N_GEN,
        halloffame=hof,
        verbose=False,
    )
    best = hof[0]
    return best, toolbox


# ----------------------------------------------------------------------
# CV
# ----------------------------------------------------------------------
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEAP symbolic regression on chemprop-anchor residual")
    print(f"   pop={POP_SIZE}  gen={N_GEN}  tourn={TOURN_K}  cxpb={CXPB}  mutpb={MUTPB}")
    print(f"   features={PHYS_KEYS}")
    print("=" * 78)

    oof_full = np.full(253, np.nan, dtype=np.float64)
    fold_rae = []
    fold_models = []
    fold_train_mse = []

    for fi, (tr_idx, va_idx) in enumerate(folds):
        ft0 = time.time()
        X_tr, X_va = X_unb[tr_idx], X_unb[va_idx]
        y_tr = resid_unb[tr_idx]
        anchor_va = oof_anchor[va_idx]
        y_va_truth = y_unb[va_idx]

        fold_seed = KF_SEED * 100 + fi
        try:
            best, tb = evolve_one_fold(X_tr, y_tr, fold_seed)
            resid_hat_va = _predict_tree(best, tb, X_va)
            tr_mse = float(best.fitness.values[0]) if best.fitness.valid \
                else float("nan")
            best_str = str(best)
        except Exception as e:
            print(f"[fold {fi}] GP failed: {e}")
            resid_hat_va = np.zeros_like(y_va_truth)
            tr_mse = float("nan")
            best_str = f"FAIL: {e}"

        final_va = anchor_va + resid_hat_va
        oof_full[va_idx] = final_va
        r = float(rae(y_va_truth, final_va))
        fold_rae.append(r)
        fold_models.append(best_str)
        fold_train_mse.append(tr_mse)
        print(f"[fold {fi}] tr={len(tr_idx)} va={len(va_idx)} "
              f"RAE={r:.4f}  tr_mse={tr_mse:.4f}  elapsed={time.time()-ft0:.1f}s")
        print(f"           best_expr = {best_str[:140]}")

    mean_rae = float(np.mean(fold_rae))
    std_rae = float(np.std(fold_rae))
    print(f"\n[cv] mean RAE = {mean_rae:.4f}  +/- {std_rae:.4f}")

    # ----------------------------------------------------------------------
    # Deploy: refit on FULL 253, predict 513 residual, add to te_anchor
    # ----------------------------------------------------------------------
    print("\n[deploy] refit on full 253 ...")
    deploy_seed = KF_SEED * 100 + 99
    best_deploy, tb_deploy = evolve_one_fold(X_unb, resid_unb, deploy_seed)
    deploy_expr = str(best_deploy)
    resid_hat_te = _predict_tree(best_deploy, tb_deploy, X_te)
    te_pred = (te_anchor + resid_hat_te).astype(np.float32)
    in_rae = float(rae(y_unb,
                       (te_anchor[unb_idx] + _predict_tree(best_deploy, tb_deploy,
                                                            X_unb))))
    te_unb_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"[deploy] expr        = {deploy_expr[:160]}")
    print(f"[deploy] in-sample RAE = {in_rae:.4f}")
    print(f"[deploy] te[unb] RAE   = {te_unb_rae:.4f}")
    print(f"[deploy] te(513) m/s   = {te_pred.mean():.3f}/{te_pred.std():.3f}")

    # ----------------------------------------------------------------------
    # Save artifacts
    # ----------------------------------------------------------------------
    pred_oof_path = PROC / f"{TAG}_pred_oof.npy"
    te_path = PROC / f"te_{TAG}.npy"
    np.save(pred_oof_path, oof_full.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")

    # ----------------------------------------------------------------------
    # Gate
    # ----------------------------------------------------------------------
    if mean_rae < GATE_PROMOTE:
        status = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        status = "MARGINAL_BEAT"
    else:
        status = "FAIL"
    print(f"\n[gate] mean_rae={mean_rae:.4f}  promote<{GATE_PROMOTE}"
          f"  marginal<{GATE_MARGINAL}  ->  {status}")

    elapsed = time.time() - t0
    payload = {
        "anchor": "nb2240_K20_mean_bag",
        "anchor_oof_path": "nb2240_mean_bag_oof_K20.npy",
        "anchor_te_path": "te_nb2240_K20.npy",
        "features": PHYS_KEYS,
        "n_unb": 253,
        "n_te": 513,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "gp_hyperparams": {
            "population": POP_SIZE,
            "generations": N_GEN,
            "tournament_size": TOURN_K,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "init_min_depth": INIT_MIN_DEPTH,
            "init_max_depth": INIT_MAX_DEPTH,
            "max_tree_height": MAX_TREE_HEIGHT,
            "crossover": "gp.cxOnePoint (subtree)",
            "mutation": "stochastic_mix(mutShrink, mutUniform random subtree)",
            "selection": "tournament",
            "binary_primitives": ["add", "sub", "mul", "pdiv"],
            "unary_primitives": ["sin", "cos", "pexp", "plog", "psqrt"],
            "ephemeral_const": "uniform(-1, 1)",
        },
        "fold_rae": fold_rae,
        "fold_train_mse": fold_train_mse,
        "fold_best_expressions": fold_models,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "deploy_expression": deploy_expr,
        "deploy_expression_len": len(deploy_expr),
        "in_sample_rae_overfit_bound": in_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "deploy_te_mean": float(te_pred.mean()),
        "deploy_te_std": float(te_pred.std()),
        "pred_oof_path": str(pred_oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(elapsed, 2),
    }
    save_summary(status, payload)
    print(f"[save] {SUMMARY_PATH}")
    print("=" * 78)
    print(f"=== {TAG} SUMMARY  status={status}  mean_rae={mean_rae:.4f}  "
          f"wall={elapsed:.1f}s ===")
    print("=" * 78)
    return {"status": status, "mean_rae": mean_rae}


if __name__ == "__main__":
    main()

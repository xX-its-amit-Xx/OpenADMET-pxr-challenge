"""nb1126 — COMBINATORIAL model/HPO/data-prep search (autonomous, resumable, cross-series gated).

Mechanically explores: feature-set x architecture x hyperparameters x data-prep x loss. Caches the featurized matrices
ONCE (C:/pxr_work/search/feats.npz) so each run is fast. Resumable: skips configs already in the results log. Each run
evaluates the next BATCH of untested configs on the HONEST metric: mean PER-SERIES RAE (cross-series, blinded-transfer
proxy) + corr-with-nb3200-error + cross-series blend delta. Logs to C:/pxr_work/search/results.jsonl; reports the best.

Usage: python nb1126_combinatorial_search.py [--build-cache] [--batch N]
"""
import os, sys, json, itertools, hashlib, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; os.makedirs(SD, exist_ok=True)
CACHE = f"{SD}/feats.npz"; LOG = f"{SD}/results.jsonl"


def build_cache():
    from src.pxr.featurize import combined, impute, morgan, rdkit_desc
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy")
    smtr = tr["smiles"].tolist(); smte = te["smiles"].to_numpy()[unb].tolist()
    print("featurizing all feature sets (one-time)...", flush=True)
    comb_tr = impute(combined(smtr)).astype(np.float32); comb_te = impute(combined(smte)).astype(np.float32)
    emb_tr = np.load(f"{P}/tr_chemprop_embed_300.npy").astype(np.float32); emb_te = np.load(f"{P}/te_chemprop_embed_300.npy")[unb].astype(np.float32)
    mor_tr = morgan(smtr).astype(np.float32); mor_te = morgan(smte).astype(np.float32)
    np.savez_compressed(CACHE, comb_tr=comb_tr, comb_te=comb_te, emb_tr=emb_tr, emb_te=emb_te,
                        mor_tr=mor_tr, mor_te=mor_te, ytr=tr["pec50"].to_numpy(), se=tr["pec50_se"].to_numpy(),
                        y=np.load(f"{P}/_audit_unblind_y.npy"), anchor=np.load(f"{P}/nb3200_pred_oof.npy"))
    print(f"cached -> {CACHE}", flush=True)


def feature_matrix(d, fset):
    parts_tr, parts_te = [], []
    m = {"combined": ("comb_tr", "comb_te"), "embed": ("emb_tr", "emb_te"), "morgan": ("mor_tr", "mor_te")}
    for tok in fset.split("+"):
        a, b = m[tok]; parts_tr.append(d[a]); parts_te.append(d[b])
    return np.hstack(parts_tr), np.hstack(parts_te)


def make_model(arch, hp):
    import lightgbm as lgb, xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler as _SS
    if arch == "lgbm":
        kw = dict(n_estimators=hp["n"], num_leaves=hp["leaves"], learning_rate=hp["lr"], subsample=hp["sub"],
                  colsample_bytree=hp["col"], reg_lambda=hp.get("l2", 0), n_jobs=4, verbose=-1)
        if hp.get("loss") == "huber": kw.update(objective="huber")
        elif hp.get("loss") == "mae": kw.update(objective="mae")
        elif hp.get("loss") == "quantile": kw.update(objective="quantile", alpha=hp.get("alpha", 0.5))
        return lgb.LGBMRegressor(**kw)
    if arch == "xgb":
        return xgb.XGBRegressor(n_estimators=hp["n"], max_depth=hp["depth"], learning_rate=hp["lr"],
                                subsample=hp["sub"], colsample_bytree=hp["col"], reg_lambda=hp.get("l2", 1), n_jobs=4)
    if arch == "cat":
        return CatBoostRegressor(iterations=hp["n"], depth=hp["depth"], learning_rate=hp["lr"],
                                 l2_leaf_reg=hp.get("l2", 3), verbose=0, random_seed=0,
                                 train_dir="C:/Temp/catboost_tmp")
    if arch == "histgb":
        return HistGradientBoostingRegressor(max_iter=hp["n"], learning_rate=hp["lr"], max_leaf_nodes=hp["leaves"],
                                             l2_regularization=hp.get("l2", 0))
    if arch == "extratrees":
        return ExtraTreesRegressor(n_estimators=hp["n"], max_depth=hp.get("depth"), n_jobs=4, random_state=0)
    if arch == "ridge":
        return Ridge(alpha=hp["alpha"])
    if arch == "enet":
        return ElasticNet(alpha=hp["alpha"], l1_ratio=hp.get("l1", 0.5), max_iter=2000)
    if arch == "mlp":   # deep tabular
        return make_pipeline(_SS(), MLPRegressor(hidden_layer_sizes=hp["hidden"], alpha=hp["alpha"],
                             learning_rate_init=hp.get("lr", 1e-3), early_stopping=True, max_iter=300, random_state=0))


def gen_configs():
    grids = {
        "lgbm": dict(n=[400, 800], leaves=[31, 64, 128], lr=[0.02, 0.04, 0.08], sub=[0.7, 0.9], col=[0.7, 0.9],
                     l2=[0, 1.0], loss=["l2", "mae", "huber"]),
        "xgb": dict(n=[400, 800], depth=[4, 6, 8], lr=[0.02, 0.04, 0.08], sub=[0.7, 0.9], col=[0.7, 0.9], l2=[1, 5]),
        "cat": dict(n=[400, 800], depth=[4, 6, 8], lr=[0.03, 0.06], l2=[3, 9]),
        "histgb": dict(n=[300, 600], leaves=[31, 63], lr=[0.03, 0.06], l2=[0, 1.0]),
        "extratrees": dict(n=[400, 800], depth=[None, 20]),
        "ridge": dict(alpha=[1, 10, 100]),
        "enet": dict(alpha=[0.01, 0.1], l1=[0.3, 0.7]),
        "mlp": dict(hidden=[(256, 64), (512, 128, 32), (1024, 256)], alpha=[1e-4, 1e-3], lr=[1e-3]),
    }
    fsets = ["combined+embed", "combined", "embed", "morgan", "morgan+embed"]
    preps = ["none", "noisy20", "noisy30", "relevant30"]
    out = []
    for fset in fsets:
        for arch, g in grids.items():
            keys = list(g);
            for vals in itertools.product(*[g[k] for k in keys]):
                hp = dict(zip(keys, vals))
                for prep in preps:
                    out.append({"fset": fset, "arch": arch, "hp": hp, "prep": prep})
    # INTERLEAVE by architecture (round-robin) so diverse archs are evaluated EARLY (enables the diverse ensemble)
    by_arch = {}
    for c in out: by_arch.setdefault(c["arch"], []).append(c)
    inter, lists = [], list(by_arch.values()); i = 0
    while any(i < len(l) for l in lists):
        for l in lists:
            if i < len(l): inter.append(l[i])
        i += 1
    return inter


def cfg_id(c): return hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build-cache", action="store_true"); ap.add_argument("--batch", type=int, default=30)
    a = ap.parse_args()
    if a.build_cache or not os.path.exists(CACHE):
        build_cache()
    d = np.load(CACHE)
    ytr, se, y, anchor = d["ytr"], d["se"], d["y"], d["anchor"]; err = y - anchor
    te = load_test(); unb = np.load(f"{P}/_audit_unblind_idx.npy")
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import Chem
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else "" for s in te["smiles"].to_numpy()[unb]]
    cte_dummy, _ = feature_matrix(d, "embed")
    series = KMeans(6, n_init=5, random_state=0).fit_predict(PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(d["emb_te"])))
    # test-relevance for prep
    from src.pxr.chem import morgan_fp_batch
    Ftr = (morgan_fp_batch(load_train().dropna(subset=["pec50"])["smiles"].tolist()).astype(np.float32) > 0).astype(np.float32)
    Fte = (morgan_fp_batch(te["smiles"].to_numpy()[unb].tolist()).astype(np.float32) > 0).astype(np.float32)
    inter = Ftr @ Fte.T; ss = Ftr.sum(1); simmax = (inter / np.clip(ss[:, None] + Fte.sum(1)[None, :] - inter, 1, None)).max(1)

    done = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    configs = gen_configs()
    todo = [c for c in configs if cfg_id(c) not in done]
    print(f"config space {len(configs)} total | done {len(done)} | remaining {len(todo)}", flush=True)

    def perseries(p): return float(np.mean([rae(y[series == k], p[series == k]) for k in range(6) if (series == k).sum() >= 5]))

    ran = 0
    with open(LOG, "a") as logf:
        for c in todo[:a.batch]:
            try:
                Xtr, Xte = feature_matrix(d, c["fset"])
                mask = np.ones(len(ytr), bool)
                if c["prep"] == "noisy20": mask = se <= np.quantile(se, 0.8)
                elif c["prep"] == "noisy30": mask = se <= np.quantile(se, 0.7)
                elif c["prep"] == "relevant30": mask = simmax >= 0.3
                m = make_model(c["arch"], c["hp"])
                if c["arch"] in ("ridge", "enet"):
                    sc = StandardScaler().fit(Xtr[mask]); m.fit(sc.transform(Xtr[mask]), ytr[mask]); p = m.predict(sc.transform(Xte))
                else:
                    m.fit(Xtr[mask], ytr[mask]); p = m.predict(Xte)
                ps_rae = perseries(p); st_rae = rae(y, p); ce = float(np.corrcoef(p, err)[0, 1])
                # cross-series blend with nb3200
                oof = anchor.copy()
                for kk in range(6):
                    tr2 = series != kk; va = series == kk
                    if va.sum() < 5: continue
                    bw, bb = 0, rae(y[tr2], anchor[tr2])
                    for w in np.linspace(0, 1, 21):
                        r = rae(y[tr2], (1 - w) * anchor[tr2] + w * p[tr2])
                        if r < bb: bb, bw = r, w
                    oof[va] = (1 - bw) * anchor[va] + bw * p[va]
                xs = float(rae(y, oof))
                rec = dict(id=cfg_id(c), **c, st_rae=float(st_rae), ps_rae=ps_rae, corr_err=ce, blend_xser=xs)
            except Exception as e:
                rec = dict(id=cfg_id(c), **c, error=str(e)[:80])
            logf.write(json.dumps(rec) + "\n"); logf.flush(); ran += 1
    # report best so far
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "blend_xser" in r]
    valid.sort(key=lambda r: r["blend_xser"])
    print(f"\nran {ran} this batch | total evaluated {len(valid)} | nb3200 anchor 0.4416")
    print("=== TOP 5 by cross-series blend (deploy metric) ===")
    for r in valid[:5]:
        print(f"  blend_xser {r['blend_xser']:.4f} | st_rae {r['st_rae']:.4f} | {r['arch']:10s} {r['fset']:14s} {r['prep']:10s} corr_err {r['corr_err']:+.3f}")
    print("=== TOP 5 by standalone per-series robustness ===")
    for r in sorted(valid, key=lambda r: r["ps_rae"])[:5]:
        print(f"  ps_rae {r['ps_rae']:.4f} | {r['arch']:10s} {r['fset']:14s} {r['prep']:10s}")


if __name__ == "__main__":
    main()

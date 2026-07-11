# smolbench

**Automated model + data optimization for *small* small-molecule datasets — with honest validation baked in.**

Give it a training CSV (`SMILES` + a numeric target) and it combinatorially searches
**featurizers × data-prep × models × hyperparameters** under **scaffold/cluster
cross-validation**, reports out-of-fold metrics, fits the best configuration on the full
training set, predicts your test set, and surfaces the small-data pitfalls that quietly
wreck QSAR models on novel chemistry.

Built from the lessons of a full OpenADMET blind-challenge campaign (≈1,800 modeling
experiments) — where the single biggest mistake was **trusting a finely-tuned ensemble
that overfit a 253-compound validation set and lost on the blind test.** smolbench refuses
to make that mistake for you: scaffold CV is the default, and it *warns you* when random CV
is lying to you.

---

## Install

```bash
pip install -e .            # core (rdkit + sklearn)
pip install -e ".[all]"     # + lightgbm / xgboost / catboost
```

## 60-second use

```python
import smolbench as sb

res = sb.benchmark(
    "train.csv", "test.csv",
    smiles_col="SMILES", target_col="pEC50",
    featurizers=["morgan", "rdkit_desc", "maccs", "erg", "atompair"],
    models=["ridge", "rf", "lgbm", "knn", "svr", "histgbm"],
    cv="scaffold",          # honest default for novel-chemistry test sets
    ensemble=True,          # blend the top-k configs
    calibrate=True,         # honest isotonic calibration (fit on OOF, never on test)
    out_dir="out/",
)

print(res.top(10))          # ranked configurations
print(sb.text_report(res))  # human-readable summary + WARNINGS
sb.make_figures(res, "out/", y_true=test_labels)   # heatmap + pred-vs-truth (if you have labels)
```

Outputs written to `out/`: `benchmark_results.csv`, `test_predictions.csv`, `insights.txt`,
`heatmap_featurizer_model.png`, `model_ranking.png`, `test_pred_vs_truth.png`.

---

## What it searches (all drop-in)

| Knob | Options |
|---|---|
| **Featurizers** (RDKit, no GPU/network) | `morgan`, `morgan_counts`, `maccs`, `avalon`, `atompair`, `torsion`, `erg`, `rdkit_desc` |
| **Data prep** | `none`, `standard`, `robust`, `pca50`, `pca200` (auto-applied to dense models) |
| **Models** | `ridge`, `elasticnet`, `rf`, `extratrees`, `histgbm`, `knn`, `svr`, `krr`, `gp` (+ `lgbm`, `xgboost`, `catboost` if installed) |
| **CV** | `scaffold` (Bemis-Murcko), `butina` (Tanimoto cluster), `random` |
| **HPO** | small, sane per-model grid (toggle with `hpo=False`) |
| **Post-processing** | top-k ensemble, honest isotonic calibration |
| **Metrics** | RAE, MAE, RMSE, R², Spearman, Kendall |

Everything is a registry — add your own featurizer or model in three lines:

```python
@sb.register_featurizer("my_fp")
def my_fp(smiles): ...            # -> np.ndarray (n, d)

sb.register_model("my_model", lambda **k: MyRegressor(**k), grid=[{"alpha": 1}])
```

## Command line

```bash
smolbench train.csv test.csv --smiles SMILES --target pEC50 --cv scaffold --out out/
smolbench train.csv test.csv --target pEC50 --truth measured_pEC50   # add metrics+figures
smolbench --list                                                     # show featurizers/models
smolbench train.csv --featurizers all --models all --target y        # kitchen sink
```

## Don't get fooled by seed luck: `stability_check`

A single CV split's RAE has real variance. Before you declare a "winner", check it clears
the noise band over many seeds (the deep-N lesson):

```python
best = res.results.iloc[0].to_dict()
st = sb.stability_check("train.csv", best, target_col="pEC50", n_seeds=15)
# -> {'rae_mean': 0.619, 'rae_std': 0.004, 'rae_ci95': [...], ...}

cmp = sb.compare_top("train.csv", res, target_col="pEC50", k=3, n_seeds=15)
# adds a 'distinguishable_from_best' column: configs within 1 std of #1 are NOT really better
```

## The small-data guardrails (why this exists)

`res.insights` auto-flags, e.g.:

- ⚠ **Small-N** (`n < 500`) → *"prefer the single most robust model over a finely-tuned
  stack — tuned ensembles overfit and lose on series-shifted test sets."*
- ⚠ **Random-CV optimism** → if random CV is ≫ better than scaffold CV, your test set is
  novel chemistry and the random number is a lie. It reports the exact gap.
- Best model family / best featurizer / recommended config, with the honest scaffold-CV RAE.

## Worked example (real, reproducible)

`examples/run_pxr.py` runs the full pipeline on the OpenADMET PXR activity data (4,392
train → 260 truly-blind test). 30 configurations, ~50 min on CPU. It auto-selected
`rdkit_desc + histgbm` by honest scaffold-CV and, on the **truly-blind** 260 held out until
the challenge ended, scored:

| smolbench auto-baseline (drop-in features only) | RAE | MAE | R² | Spearman |
|---|---|---|---|---|
| best single config (`rdkit_desc + histgbm`) — scaffold-CV | 0.619 | 0.532 | 0.57 | 0.71 |
| **top-5 ensemble + isotonic — on the blind 260** | **0.747** | **0.528** | 0.40 | 0.73 |

That MAE 0.53 comes from **nothing but RDKit fingerprints + sklearn**, fully automated.
(Our full challenge pipeline — adding Boltz cofolding, a single-concentration functional
screen, and a Chemprop GNN — reached MAE 0.47, but that took ~1,800 experiments; smolbench
gets you 90% of the way in one function call.) Generated figures live in `examples/pxr_out/`:

![featurizer x model heatmap](examples/pxr_out/heatmap_featurizer_model.png)
![blind-set predictions](examples/pxr_out/test_pred_vs_truth.png)

## Drop into DeepChem

`smolbench.benchmark` accepts either CSV paths or pandas DataFrames and returns plain
numpy/pandas — so it slots into a DeepChem pipeline as a model/featurizer selection step,
or runs standalone. The featurizer and model registries mirror DeepChem's plug-in style.

A **DeepChem-native** version (using DeepChem's own featurizers/splitters, no external
dep) has been contributed upstream:
**[deepchem/deepchem#5054](https://github.com/deepchem/deepchem/pull/5054)** — see
[`deepchem_integration/`](deepchem_integration/).

## PyPI

Package builds cleanly (`python -m build`; `twine check` passes). Publish with:

```bash
python -m build
twine upload dist/*        # requires a PyPI API token
```

## License
MIT.

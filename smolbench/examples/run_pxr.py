"""Demo: run smolbench on the OpenADMET PXR activity data (4392 train -> 260 blind test)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
import smolbench as sb

REPO = "D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge"
crc = pd.read_csv(f"{REPO}/data/raw/pxr-challenge_TRAIN.csv")
p1 = pd.read_csv(f"{REPO}/data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
p2 = pd.read_csv(f"{REPO}/data/raw/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")

def col(df, *c):
    for x in c:
        for k in df.columns:
            if k.lower() == x.lower():
                return k
smi_c = col(crc, "SMILES", "smiles"); y_c = col(crc, "pEC50")
train = pd.concat([
    crc[[smi_c, y_c]].rename(columns={smi_c: "SMILES", y_c: "pEC50"}),
    p1[["SMILES", "pEC50"]],
], ignore_index=True).dropna()
test = p2[["SMILES", "pEC50"]].rename(columns={"pEC50": "truth"})

print(f"train {len(train)}  test {len(test)}")
res = sb.benchmark(
    train, test[["SMILES"]], smiles_col="SMILES", target_col="pEC50",
    featurizers=["morgan", "rdkit_desc", "maccs", "erg", "atompair"],
    models=["ridge", "rf", "lgbm", "knn", "svr", "histgbm"],
    preps=("standard",), cv="scaffold", n_folds=5,
    hpo=True, ensemble=True, top_k=5, calibrate=True,
    out_dir=f"{REPO}/smolbench/examples/pxr_out",
)
print("\n" + sb.text_report(res))
# score on the real 260 truth (post-hoc)
from smolbench.metrics import all_metrics
m = all_metrics(test["truth"].values, res.test_predictions)
print("\nHOLD-OUT 260 (blind) metrics of smolbench's auto-selected ensemble:")
print("  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
sb.make_figures(res, f"{REPO}/smolbench/examples/pxr_out", y_true=test["truth"].values, target_name="pEC50")
print("figures + results written to smolbench/examples/pxr_out/")

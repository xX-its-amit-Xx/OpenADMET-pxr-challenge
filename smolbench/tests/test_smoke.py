"""Fast smoke test: tiny dataset, few configs — verifies the pipeline end-to-end."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
import smolbench as sb

# 60 simple molecules with a target correlated to heavy-atom count
smis = ["C" * (i % 8 + 1) for i in range(30)] + [
    "c1ccccc1", "CCO", "CCN", "CCC(=O)O", "c1ccncc1", "CC(C)C", "CCCCCC", "c1ccc(O)cc1",
    "CC(=O)Nc1ccccc1", "OCC(O)CO", "c1ccc2ccccc2c1", "CN1CCCC1", "FC(F)F", "ClCCl",
    "CC(C)(C)O", "c1ccsc1", "c1cc[nH]c1", "CCOCC", "CC#N", "O=C(O)c1ccccc1",
    "Cc1ccccc1C", "CCCCO", "NCCO", "c1ccc(Cl)cc1", "CC(N)C(=O)O", "OCCO",
    "c1ccc(N)cc1", "CCCCCCCC", "CC(C)Cc1ccccc1", "COc1ccccc1"]
from rdkit import Chem
y = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() + np.random.RandomState(i).randn()
              for i, s in enumerate(smis)], float)
df = pd.DataFrame({"SMILES": smis, "y": y})
test = df.iloc[:10][["SMILES"]]

res = sb.benchmark(df.iloc[10:], test, smiles_col="SMILES", target_col="y",
                   featurizers=["morgan", "rdkit_desc"], models=["ridge", "rf"],
                   cv="random", n_folds=3, hpo=False, ensemble=True, calibrate=False, verbose=False)
assert len(res.results) > 0, "no configs evaluated"
assert res.test_predictions is not None and len(res.test_predictions) == 10
assert "RAE" in res.results.columns
print("SMOKE OK — configs:", len(res.results), "| best RAE:", round(res.results.iloc[0]["RAE"], 3))
print("insights:", res.insights[0][:70])
print(sb.available_featurizers()[:5], "...", sb.available_models()[:5], "...")

"""
nb2097 Kaggle v8 runner — chemprop 2.1.x retrain
Run via Kaggle dataset `knowledgegraphlover/pxr-challenge-data`
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "chemprop==2.1.0", "lightning>=2.2", "rdkit"])
# ... full training loop omitted for stub; will be filled when v7 fails
print("nb2097 v8 dispatcher stub — fill in when v7 verdict known")

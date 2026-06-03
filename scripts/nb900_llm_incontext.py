"""nb900 -- LLM in-context PXR pEC50 prediction.

For each of the 513 test compounds, retrieve the 20 most similar training
compounds by Morgan/Tanimoto, build a prompt listing their pEC50 values,
and ask Claude (haiku-4-5) to predict the test compound's pEC50.

Each compound is queried 5 times across a small temperature ladder; the
mean of the parsed floats is the final prediction. Predictions are batched
8-wide via asyncio to keep wall-clock around ~1 hour.

Outputs:
  - C:/pxr_artifacts/te_nb900.npy           # (513,) float32 mean prediction
  - submissions/nb900_llm_incontext.csv     # 513-row submission
  - C:/pxr_artifacts/nb900_raw.json         # raw per-call responses (debug)

If ANTHROPIC_API_KEY is unset OR the `anthropic` SDK is unavailable, the
script falls back to writing a prompt manifest at
  C:/pxr_artifacts/nb900_prompts.jsonl
that can be processed externally.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pxr.eval import rae  # noqa: E402
from pxr.paths import DATA_PROCESSED, SUBMISSIONS  # noqa: E402

ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)

MODEL = "claude-haiku-4-5-20251001"
K_NEIGHBORS = 20
N_REPLICATES = 5
TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]
BATCH = 8
MAX_TOKENS = 64

FLOAT_RE = re.compile(r"-?\d+\.\d+|-?\d+")


def fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def top_k(test_fp, train_fps, train_y, train_smi, k=K_NEIGHBORS):
    sims = np.array(DataStructs.BulkTanimotoSimilarity(test_fp, train_fps))
    idx = np.argsort(-sims)[:k]
    return [(train_smi[i], float(train_y[i]), float(sims[i])) for i in idx]


def build_prompt(test_smi: str, neighbors) -> str:
    lines = [f"  - SMILES: {s}  pEC50 = {y:.2f}  (Tanimoto={t:.2f})"
             for s, y, t in neighbors]
    return (
        "You are predicting human PXR (pregnane X receptor) agonist pEC50 "
        "(–log10 EC50 in molarity; higher = more potent).\n\n"
        f"Test SMILES: {test_smi}\n\n"
        f"Here are the {len(neighbors)} most similar training compounds with "
        "measured PXR pEC50 (sorted by descending Morgan/Tanimoto similarity):\n"
        + "\n".join(lines)
        + "\n\nPredict the test compound's pEC50. Respond with ONLY a single "
        "decimal number (e.g. 5.47). No words, no units, no explanation."
    )


def parse_float(text: str) -> float | None:
    if not text:
        return None
    m = FLOAT_RE.search(text)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    if not np.isfinite(v) or v < 0 or v > 12:
        return None
    return v


async def call_one(client, prompt: str, temperature: float) -> str:
    msg = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text if msg.content else ""


async def predict_compound(client, prompt: str, sem: asyncio.Semaphore):
    async def one(t):
        async with sem:
            for attempt in range(3):
                try:
                    txt = await call_one(client, prompt, t)
                    return txt
                except Exception as e:
                    if attempt == 2:
                        return f"ERR:{e}"
                    await asyncio.sleep(2 ** attempt)
    return await asyncio.gather(*[one(t) for t in TEMPERATURES])


async def run_with_api(prompts, fallback):
    import anthropic
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(BATCH)
    raw_all = []
    preds = np.full(len(prompts), np.nan)
    for i, p in enumerate(prompts):
        texts = await predict_compound(client, p, sem)
        vals = [parse_float(t) for t in texts]
        vals = [v for v in vals if v is not None]
        preds[i] = float(np.mean(vals)) if vals else fallback[i]
        raw_all.append({"i": i, "texts": texts, "parsed": vals, "pred": float(preds[i])})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(prompts)}  last_pred={preds[i]:.3f}", flush=True)
    return preds, raw_all


def main():
    print("Loading data...", flush=True)
    tr = pd.read_csv(REPO / "data/raw/pxr-challenge_TRAIN.csv")
    tr = tr.dropna(subset=["pEC50", "SMILES"]).reset_index(drop=True)
    te = pd.read_csv(REPO / "data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(REPO / "data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")

    print(f"Train: {len(tr)}  Test: {len(te)}  Unblind: {len(unb)}", flush=True)

    print("Fingerprinting train...", flush=True)
    tr_fps = [fp(s) for s in tr["SMILES"]]
    keep = [i for i, f in enumerate(tr_fps) if f is not None]
    tr_fps = [tr_fps[i] for i in keep]
    tr_y = tr["pEC50"].values[keep]
    tr_smi = tr["SMILES"].values[keep]

    print("Fingerprinting test + building prompts...", flush=True)
    te_fps = [fp(s) for s in te["SMILES"]]
    prompts = []
    knn_fallback = np.full(len(te), float(np.mean(tr_y)))
    for i, f in enumerate(te_fps):
        if f is None:
            prompts.append("")
            continue
        nbrs = top_k(f, tr_fps, tr_y, tr_smi)
        prompts.append(build_prompt(te["SMILES"].iloc[i], nbrs))
        # Similarity-weighted kNN fallback for unparsable LLM replies
        sims = np.array([n[2] for n in nbrs])
        ys = np.array([n[1] for n in nbrs])
        if sims.sum() > 0:
            knn_fallback[i] = float((sims * ys).sum() / sims.sum())

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        import anthropic  # noqa: F401
        have_sdk = True
    except ImportError:
        have_sdk = False

    if not (have_key and have_sdk):
        manifest = ART / "nb900_prompts.jsonl"
        with manifest.open("w", encoding="utf-8") as fh:
            for i, (name, smi, p) in enumerate(zip(
                te["Molecule Name"], te["SMILES"], prompts
            )):
                fh.write(json.dumps({
                    "i": i, "name": name, "smiles": smi,
                    "prompt": p, "model": MODEL,
                    "temperatures": TEMPERATURES,
                    "knn_fallback": float(knn_fallback[i]),
                }) + "\n")
        reason = []
        if not have_key:
            reason.append("ANTHROPIC_API_KEY not set")
        if not have_sdk:
            reason.append("anthropic SDK not installed")
        print(f"Skipping API calls ({', '.join(reason)}). "
              f"Wrote manifest: {manifest}", flush=True)
        # Save kNN fallback so downstream code has *something* at the artifact path.
        np.save(ART / "te_nb900.npy", knn_fallback.astype(np.float32))
        print(f"Saved kNN fallback as te_nb900.npy "
              f"(mean={knn_fallback.mean():.3f}, std={knn_fallback.std():.3f})",
              flush=True)
        return

    print(f"Running {len(prompts)} x {N_REPLICATES} = "
          f"{len(prompts)*N_REPLICATES} LLM calls (batch={BATCH})...", flush=True)
    preds, raw_all = asyncio.run(run_with_api(prompts, knn_fallback))

    preds = preds.astype(np.float32)
    np.save(ART / "te_nb900.npy", preds)
    with (ART / "nb900_raw.json").open("w", encoding="utf-8") as fh:
        json.dump(raw_all, fh)

    # Evaluate on unblind subset
    name_to_idx = {n: i for i, n in enumerate(te["Molecule Name"])}
    mask = [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    unb_y = unb.loc[unb["Molecule Name"].isin(name_to_idx), "pEC50"].values
    in_rae = float(rae(unb_y, preds[mask])) if len(mask) else float("nan")
    print(f"Unblind in_RAE: {in_rae:.4f}  "
          f"(n={len(mask)}, mean={preds.mean():.3f}, std={preds.std():.3f})",
          flush=True)

    out = SUBMISSIONS / "nb900_llm_incontext.csv"
    pd.DataFrame({
        "Molecule Name": te["Molecule Name"],
        "SMILES": te["SMILES"],
        "pEC50": preds,
    }).to_csv(out, index=False)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()

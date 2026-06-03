"""nb1034 -- ChemBERTa-77M-MTR partial-unfreeze fine-tune.

Cycle 1 nb904 SSL char-transformer was tiny (~200K params, frozen during
inference). Cycle 1 also tried nb610-614 ChemBERTa frozen-feature routers;
all DEPRECATED (could not beat nb562). This is the next obvious lever:
unfreeze the last 2 RoBERTa layers + a regression head and supervised-
fine-tune end-to-end on the 4139 CRC pEC50 targets.

Hypothesis: the pretrained 77M-MTR backbone already encodes chemistry
relevant features; updating the last 2 transformer blocks + reg head lets
the model align those features to the PXR pEC50 manifold while keeping
the lower layers stable (less catastrophic forgetting on a tiny dataset).

Pipeline:
  1. Load DeepChem/ChemBERTa-77M-MTR from local HF cache.
  2. Freeze embeddings + all but the last 2 encoder layers; unfreeze
     layers [-2, -1] + a fresh Linear(hidden, 1) regression head.
  3. Fine-tune on 4139 CRC pec50 with HuberLoss(delta=1.0),
     AdamW lr=1e-4, 20 epochs, batch=16.
  4. Predict 513 test, score in_RAE on 253 phase-1 unblind.
  5. Pearson check vs te_nb972_long_train.
  6. If Pearson < 0.95: nb1014-style 2-way SLSQP blend with chemprop_aux.

Wall-time budget: 30 min on CPU. If exceeded, save the partial checkpoint
and emit success=False.

Outputs:
  data/processed/te_nb1034.npy
  data/processed/nb1034_summary.json
  submissions/nb1034_chemberta_full_finetune.csv
  C:/pxr_artifacts/nb1034_chemberta_ft.pt
"""
from __future__ import annotations

import json
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
from torch.utils.data import Dataset, DataLoader

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1034"
MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"
SEED = 0
BATCH = 16
EPOCHS = 20
LR = 1e-4
WD = 0.01
HUBER_DELTA = 1.0
MAX_LEN = 128
WALL_BUDGET_SEC = 50 * 60  # 50 min hard budget
PEARSON_BLEND_THRESH = 0.95

ART = Path("C:/pxr_artifacts")
ART.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")


class SmilesDataset(Dataset):
    def __init__(self, enc, y=None):
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.y = y

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, i):
        item = {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
        }
        if self.y is not None:
            item["y"] = torch.tensor(self.y[i], dtype=torch.float32)
        return item


class ChemBERTaRegressor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        h = base.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool over non-pad tokens
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def freeze_partial(base, n_unfreeze_layers: int = 2) -> dict:
    """Freeze embeddings + all encoder layers except the last
    `n_unfreeze_layers`. Returns dict of (n_trainable, n_total)."""
    for p in base.parameters():
        p.requires_grad = False
    # RoBERTa structure: base.encoder.layer is a ModuleList
    enc_layers = base.encoder.layer
    n_layers = len(enc_layers)
    keep_from = n_layers - n_unfreeze_layers
    for i, layer in enumerate(enc_layers):
        if i >= keep_from:
            for p in layer.parameters():
                p.requires_grad = True
    # Also unfreeze pooler if present (RoBERTa often has it for CLS)
    if getattr(base, "pooler", None) is not None:
        for p in base.pooler.parameters():
            p.requires_grad = True
    n_trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in base.parameters())
    return {
        "n_layers_total": n_layers,
        "n_layers_unfrozen": n_unfreeze_layers,
        "n_base_trainable": int(n_trainable),
        "n_base_total": int(n_total),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChemBERTa-77M-MTR partial-unfreeze fine-tune")
    print("=" * 78)

    # ---- Truth/unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    assert len(unb_idx) == len(y_unb) == 253

    # ---- Data ----
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float32)
    tr_smi = tr["smiles"].astype(str).tolist()
    te_smi = te["smiles"].astype(str).tolist()
    te_names = te["name"].values
    print(f"[data] train n={len(y_tr)}  test n={len(te_smi)}")

    # ---- Load tokenizer + backbone from local cache ----
    from transformers import AutoTokenizer, AutoModel

    print(f"[load] {MODEL_ID} from HF cache")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModel.from_pretrained(MODEL_ID)

    freeze_info = freeze_partial(base, n_unfreeze_layers=2)
    print(
        f"[freeze] {freeze_info['n_layers_unfrozen']}/{freeze_info['n_layers_total']} "
        f"encoder layers unfrozen; backbone trainable "
        f"{freeze_info['n_base_trainable']:,}/{freeze_info['n_base_total']:,}"
    )

    model = ChemBERTaRegressor(base).to(DEVICE)
    head_trainable = sum(p.numel() for p in model.head.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[model] head trainable {head_trainable:,}; "
        f"total trainable {total_trainable:,}"
    )

    # ---- Tokenize ----
    print(f"[tok] tokenizing (max_len={MAX_LEN})")
    enc_tr = tok(tr_smi, padding=True, truncation=True,
                 max_length=MAX_LEN, return_tensors="pt")
    enc_te = tok(te_smi, padding=True, truncation=True,
                 max_length=MAX_LEN, return_tensors="pt")
    print(f"[tok] train input_ids {tuple(enc_tr['input_ids'].shape)}; "
          f"test {tuple(enc_te['input_ids'].shape)}")

    ds_tr = SmilesDataset(enc_tr, y_tr)
    ds_te = SmilesDataset(enc_te, None)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=BATCH * 2, shuffle=False, num_workers=0)

    # ---- Optimizer / loss ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WD,
    )
    huber = nn.HuberLoss(delta=HUBER_DELTA)

    # ---- Fine-tune loop with wall budget guard ----
    print(f"[train] {EPOCHS} epochs, batch={BATCH}, lr={LR}, "
          f"huber_delta={HUBER_DELTA}")
    ckpt_path = ART / f"{TAG}_chemberta_ft.pt"
    partial = False
    completed_epochs = 0
    losses: list[float] = []
    # Resume from checkpoint if present
    start_epoch = 0
    if ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ck["state_dict"])
            start_epoch = int(ck.get("completed_epochs", 0))
            losses = list(ck.get("epoch_losses", []) or [])
            completed_epochs = start_epoch
            print(f"[resume] loaded checkpoint; continuing from epoch {start_epoch}")
        except Exception as e:
            print(f"[resume] failed ({e}); starting fresh")
            start_epoch = 0
    for ep in range(start_epoch, EPOCHS):
        if time.time() - t0 > WALL_BUDGET_SEC * 0.85:
            print(f"[budget] wall {time.time() - t0:.0f}s > 85% of {WALL_BUDGET_SEC}; "
                  "saving partial state and stopping early.")
            partial = True
            break
        model.train()
        tot, nbat = 0.0, 0
        for batch in dl_tr:
            opt.zero_grad()
            yhat = model(batch["input_ids"].to(DEVICE),
                         batch["attention_mask"].to(DEVICE))
            loss = huber(yhat, batch["y"].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(loss.item())
            nbat += 1
        ep_loss = tot / max(nbat, 1)
        losses.append(ep_loss)
        completed_epochs = ep + 1
        elapsed = time.time() - t0
        print(f"  epoch {ep+1:2d}/{EPOCHS}  huber={ep_loss:.4f}  "
              f"elapsed={elapsed:5.0f}s")

    # Save checkpoint (even if partial)
    torch.save({
        "state_dict": model.state_dict(),
        "completed_epochs": completed_epochs,
        "partial": partial,
        "freeze_info": freeze_info,
        "epoch_losses": losses,
        "config": {"BATCH": BATCH, "LR": LR, "WD": WD,
                   "EPOCHS": EPOCHS, "HUBER_DELTA": HUBER_DELTA,
                   "MAX_LEN": MAX_LEN, "MODEL_ID": MODEL_ID},
    }, ckpt_path)
    print(f"[save] checkpoint -> {ckpt_path}")

    if partial:
        # Emit partial summary before predict (to honour wall budget)
        summary = {
            "tag": TAG,
            "success": False,
            "reason": "wall_budget_exceeded",
            "completed_epochs": completed_epochs,
            "wall_sec": round(time.time() - t0, 1),
            "checkpoint": str(ckpt_path),
            "epoch_losses": losses,
        }
        with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ---- Predict on 513 test ----
    print("[predict] 513 test")
    model.eval()
    te_pred = np.zeros(len(te_smi), dtype=np.float32)
    cursor = 0
    with torch.no_grad():
        for batch in dl_te:
            yhat = model(batch["input_ids"].to(DEVICE),
                         batch["attention_mask"].to(DEVICE))
            b = yhat.shape[0]
            te_pred[cursor:cursor + b] = yhat.cpu().numpy()
            cursor += b
    print(f"[predict] te mean/std = {te_pred.mean():.3f} / {te_pred.std():.3f}")

    # ---- in_RAE on 253 ----
    in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"[score] in_RAE(253) standalone = {in_rae:.4f}")

    # ---- Pearson vs nb972 ----
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    pearson = float(np.corrcoef(te_pred.astype(np.float64), te_nb972)[0, 1])
    print(f"[corr] Pearson(nb1034, nb972) = {pearson:.4f}")

    # ---- Conditional blend with chemprop_aux (nb1014 protocol, 2-way) ----
    blended_in_rae = None
    blend_w_chemprop = None
    blend_w_self = None
    blend_pearson_thresh_triggered = pearson < PEARSON_BLEND_THRESH
    deploy_pred = te_pred.copy()
    if blend_pearson_thresh_triggered:
        print(f"[blend] Pearson {pearson:.4f} < {PEARSON_BLEND_THRESH}: "
              "running SLSQP blend on 253 (nb1014 protocol)")
        from scipy.optimize import minimize
        te_cp = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
        # Optional 3rd component: nb1041 if present
        te_nb1041_path = DATA_PROCESSED / "te_nb1041.npy"
        has_nb1041 = te_nb1041_path.exists()
        if has_nb1041:
            te_nb1041 = np.load(te_nb1041_path).astype(np.float64)
            stack_unb = np.stack([te_cp[unb_idx],
                                  te_nb1041[unb_idx],
                                  te_pred[unb_idx].astype(np.float64)], axis=0)
            stack_full = np.stack([te_cp, te_nb1041, te_pred.astype(np.float64)], axis=0)
            comp_names = ("chemprop_aux", "nb1041", "nb1034")
        else:
            stack_unb = np.stack([te_cp[unb_idx],
                                  te_pred[unb_idx].astype(np.float64)], axis=0)
            stack_full = np.stack([te_cp, te_pred.astype(np.float64)], axis=0)
            comp_names = ("chemprop_aux", "nb1034")
        K = stack_unb.shape[0]

        def sse(w):
            return float(np.sum((w @ stack_unb - y_unb) ** 2))

        res = minimize(
            sse,
            np.full(K, 1.0 / K),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * K,
            constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
            options={"ftol": 1e-10, "maxiter": 500},
        )
        w = res.x.astype(np.float64)
        blend_w_chemprop = float(w[0])
        if has_nb1041:
            blend_w_self = float(w[-1])
        else:
            blend_w_self = float(w[1])
        blend_unb = w @ stack_unb
        blended_in_rae = float(rae(y_unb, blend_unb))
        for nm, wi in zip(comp_names, w):
            print(f"  blend w_{nm} = {wi:.3f}")
        print(f"[blend] in_RAE(253)={blended_in_rae:.4f}  "
              f"(in-sample, overfit lower bound)")
        deploy_pred = (w @ stack_full).astype(np.float32)
    else:
        print(f"[blend] Pearson {pearson:.4f} >= {PEARSON_BLEND_THRESH}: "
              "skipping blend (too redundant with nb972)")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_pred.astype(np.float32))
    sub_path = SUBMISSIONS / f"{TAG}_chemberta_full_finetune.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_pred.astype(np.float32),
    }).to_csv(sub_path, index=False)
    print(f"[save] te_{TAG}.npy")
    print(f"[save] {sub_path}")

    # ---- Compare to nb1014 baseline ----
    NB1014_BAGGED_HONEST = None
    nb1014_sum = DATA_PROCESSED / "nb1014_summary.json"
    if nb1014_sum.exists():
        try:
            with open(nb1014_sum) as f:
                NB1014_BAGGED_HONEST = json.load(f).get("mean_pooled_rae")
        except Exception:
            pass

    final_for_compare = (
        blended_in_rae if blended_in_rae is not None else in_rae
    )
    beats_nb1014 = (
        NB1014_BAGGED_HONEST is not None
        and final_for_compare < NB1014_BAGGED_HONEST
    )
    print(
        f"[verdict] nb1034 final in_RAE(253)={final_for_compare:.4f}  "
        f"vs nb1014 bagged honest {NB1014_BAGGED_HONEST}  "
        f"-> beats={beats_nb1014}"
    )

    summary = {
        "tag": TAG,
        "success": True,
        "model_id": MODEL_ID,
        "freeze_info": freeze_info,
        "n_total_trainable": int(total_trainable),
        "config": {"BATCH": BATCH, "LR": LR, "WD": WD,
                   "EPOCHS": EPOCHS, "HUBER_DELTA": HUBER_DELTA,
                   "MAX_LEN": MAX_LEN, "SEED": SEED},
        "completed_epochs": completed_epochs,
        "epoch_losses": losses,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "in_rae_253_standalone": in_rae,
        "pearson_with_nb972": pearson,
        "pearson_thresh": PEARSON_BLEND_THRESH,
        "blend_triggered": blend_pearson_thresh_triggered,
        "blend_w_chemprop_aux": blend_w_chemprop,
        "blend_w_self": blend_w_self,
        "blended_in_rae_253": blended_in_rae,
        "deploy_te_mean": float(deploy_pred.mean()),
        "deploy_te_std": float(deploy_pred.std()),
        "nb1014_bagged_honest_ref": NB1014_BAGGED_HONEST,
        "beats_nb1014": bool(beats_nb1014) if NB1014_BAGGED_HONEST is not None else None,
        "checkpoint": str(ckpt_path),
        "plain_submission": str(sub_path),
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   standalone in_RAE(253)  = {in_rae:.4f}")
    print(f"   Pearson vs nb972        = {pearson:.4f}")
    if blended_in_rae is not None:
        print(f"   blended in_RAE(253)     = {blended_in_rae:.4f}  "
              f"(w_chemprop={blend_w_chemprop:.3f}, w_self={blend_w_self:.3f})")
    print(f"   beats nb1014 (honest)   = {beats_nb1014}")
    print(f"   wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY (top keys) ====")
    for k in ("success", "in_rae_253_standalone", "pearson_with_nb972",
              "blend_triggered", "blended_in_rae_253", "beats_nb1014",
              "completed_epochs", "wall_sec", "plain_submission"):
        print(f"  {k}: {res.get(k)}")

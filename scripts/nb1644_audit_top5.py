"""nb1644_audit_top5.py -- Integrity audit on top-5 PRE-unblind candidates.

For each candidate:
  1. Verify te_NNNN.npy exists.
  2. Verify CSV exists and has 513 rows.
  3. Compute te_RAE on 253 unblind labels.
  4. SHA256 of te[unb_idx] vs y_unb -- match = LEAK.
  5. Pearson(te[unb_idx], y_unb) -- ~1.0 + delta<0.05 = LEAK suspicion.
  6. Print integrity report.
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
SUBS = ROOT / "submissions"

UNB_IDX = np.load(DATA / "_audit_unblind_idx.npy")
UNB_Y   = np.load(DATA / "_audit_unblind_y.npy")

CANDIDATES = [
    ("nb1583",        "te_nb1583.npy",         "nb1583_deploy_nb1571.csv"),
    ("nb1570_mean",   "te_nb1570_mean.npy",    "nb1570_deploy_nb1561_mean.csv"),
    ("nb1570_median", "te_nb1570_median.npy",  "nb1570_deploy_nb1561_median.csv"),
    ("nb1480",        "te_nb1480.npy",         "nb1480_deploy_nb1472.csv"),
    ("nb1490_mean",   "te_nb1490_mean.npy",    "nb1490_deploy_nb1482_mean.csv"),
]

def rae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - y_true.mean()))
    return float(num / den) if den > 0 else float("nan")

def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()

def pearson(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

print(f"=== nb1644 top-5 PRE-unblind integrity audit ===")
print(f"unb_idx shape: {UNB_IDX.shape}  y_unb shape: {UNB_Y.shape}")
print(f"y_unb sha256: {sha256_arr(UNB_Y)[:16]}...")
print()

results = []
for name, te_name, csv_name in CANDIDATES:
    te_path = DATA / te_name
    csv_path = SUBS / csv_name
    rec = {"candidate": name, "te_file": te_name, "csv_file": csv_name}

    # 1. te npy
    rec["te_exists"] = te_path.exists()
    if not rec["te_exists"]:
        rec["status"] = "MISSING_TE"
        results.append(rec); continue
    te = np.load(te_path)
    rec["te_shape"] = te.shape
    if te.shape[0] != 513:
        rec["status"] = f"BAD_TE_LEN_{te.shape[0]}"
        results.append(rec); continue

    # 2. csv
    rec["csv_exists"] = csv_path.exists()
    if not rec["csv_exists"]:
        rec["status"] = "MISSING_CSV"
        results.append(rec); continue
    df = pd.read_csv(csv_path)
    rec["csv_rows"] = len(df)
    if len(df) != 513:
        rec["status"] = f"BAD_CSV_LEN_{len(df)}"
        results.append(rec); continue

    # 3. te[unb_idx] vs y_unb
    te_unb = te[UNB_IDX]
    rec["te_unb_rae"] = round(rae(UNB_Y, te_unb), 4)

    # 4. SHA256 leak check
    sha_te = sha256_arr(te_unb)
    sha_y  = sha256_arr(UNB_Y)
    rec["te_unb_sha"] = sha_te[:16]
    rec["sha_match"]  = (sha_te == sha_y)

    # 5. Pearson + max abs diff (loose leak signature)
    rec["pearson"]    = round(pearson(te_unb, UNB_Y), 4)
    rec["max_abs_d"]  = round(float(np.max(np.abs(te_unb - UNB_Y))), 4)
    rec["mean_abs_d"] = round(float(np.mean(np.abs(te_unb - UNB_Y))), 4)

    # 6. csv-vs-te alignment
    pec_col = None
    for c in df.columns:
        if c.lower() in ("pec50", "predicted_pec50", "ypred", "pred"):
            pec_col = c; break
    if pec_col is not None:
        csv_pred = df[pec_col].to_numpy(dtype=float)
        rec["csv_te_match"] = bool(np.allclose(csv_pred, te, atol=1e-6))
    else:
        rec["csv_te_match"] = None

    # 7. pred_oof presence (LB honest signal)
    nb_base = name.split("_")[0]
    pred_oof_path = DATA / f"{nb_base}_pred_oof.npy"
    rec["pred_oof_exists"] = pred_oof_path.exists()
    if pred_oof_path.exists():
        try:
            p = np.load(pred_oof_path)
            if p.shape[0] == 253:
                rec["pred_oof_rae"] = round(rae(UNB_Y, p), 4)
            elif p.shape[0] == 513:
                rec["pred_oof_rae"] = round(rae(UNB_Y, p[UNB_IDX]), 4)
            else:
                rec["pred_oof_rae"] = None
        except Exception as e:
            rec["pred_oof_rae"] = f"ERR:{e}"
    else:
        rec["pred_oof_rae"] = None

    # Verdict
    leak_flags = []
    if rec["sha_match"]:
        leak_flags.append("SHA256_MATCH")
    if rec["pearson"] is not None and rec["pearson"] > 0.999 and rec["te_unb_rae"] < 0.05:
        leak_flags.append("NEAR_PERFECT_FIT")
    if rec["te_unb_rae"] < 0.10:
        leak_flags.append("SUSPICIOUSLY_LOW_RAE")
    rec["leak_flags"] = leak_flags
    rec["status"] = "LEAK" if leak_flags else "OK"
    results.append(rec)

# Print
print(f"{'candidate':<16} {'status':<8} {'te_unb_rae':>10} {'pred_oof':>10} {'pearson':>8} {'sha_match':>10} {'leak_flags'}")
print("-"*120)
for r in results:
    rae_v = r.get("te_unb_rae", "n/a")
    oof_v = r.get("pred_oof_rae", None)
    oof_s = f"{oof_v:.4f}" if isinstance(oof_v, (int, float)) else str(oof_v)
    pe    = r.get("pearson", "n/a")
    sm    = r.get("sha_match", "n/a")
    lf    = ",".join(r.get("leak_flags", [])) or "-"
    print(f"{r['candidate']:<16} {r.get('status','?'):<8} {rae_v:>10} {oof_s:>10} {pe:>8} {str(sm):>10} {lf}")

print()
print("=== Detail ===")
for r in results:
    print(r)

# Recommendation
print()
print("=== RECOMMENDATION ===")
ok = [r for r in results if r.get("status") == "OK"]
ok.sort(key=lambda x: x.get("te_unb_rae", 1e9))
if not ok:
    print("ALL FAILED -- DO NOT SUBMIT")
else:
    print(f"All {len(ok)}/{len(results)} candidates clean (no leak signatures).")
    print(f"Ranked by te_unb_rae (in-sample, expect optimism):")
    for rk, r in enumerate(ok, 1):
        print(f"  #{rk}  {r['candidate']:<16}  te_unb_rae={r['te_unb_rae']}  csv_te_match={r.get('csv_te_match')}")
    best = ok[0]
    print(f"\nBest in-sample: {best['candidate']} ({best['te_unb_rae']})")
    print(f"Note: te_unb_rae is in-sample (deploy refit incl 253 unblind). For LB-honest, use pred_oof.")

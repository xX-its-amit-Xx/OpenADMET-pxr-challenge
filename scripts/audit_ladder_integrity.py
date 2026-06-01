"""audit_ladder_integrity.py -- Pre-submission integrity audit.

For every PRIMARY entry in auto_submit_ladder.LADDER:
  1. Load submissions/<file> -> pec50 column
  2. Compute te_RAE on the 253 unblind labels (_audit_unblind_idx.npy / _audit_unblind_y.npy)
  3. Compare to the stored pred_oof RAE (data/processed/<nbXXX>_pred_oof.npy)
     - If pred_oof missing, fall back to the "claimed" RAE parsed from the LADDER note
  4. If |te_RAE - pred_oof_RAE| > 0.05, mark INTEGRITY=FAIL (demote to DEPRECATED)
  5. Print final PRIMARY top-5 ranked by te_RAE_on_unblind.

Outputs:
  - prints audit table to stdout
  - rewrites scripts/auto_submit_ladder.py demoting any FAIL entries
"""
import os, sys, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxr.paths import DATA_PROCESSED, SUBMISSIONS
sys.path.insert(0, str(ROOT / "scripts"))
from auto_submit_ladder import LADDER  # type: ignore

UNB_IDX = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
UNB_Y   = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")

LADDER_PATH = ROOT / "scripts" / "auto_submit_ladder.py"
TOL = 0.05

def rae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - y_true.mean()))
    return float(num / den) if den > 0 else float("nan")

def parse_claimed_rae(note: str):
    # Look for "RAE 0.xxxx" or "0.xxxx)" patterns
    m = re.search(r"RAE\s+(0\.\d{3,4})", note)
    if m: return float(m.group(1))
    m = re.search(r"(0\.\d{3,4})\b", note)
    if m: return float(m.group(1))
    return None

def nb_id(filename: str):
    m = re.match(r"(nb\d+)", filename)
    return m.group(1) if m else None

def audit():
    rows = []
    for fn, note in LADDER:
        is_primary_active = not (note.startswith("DEPRECATED") or note.startswith("DIVERSITY") or
                                  note.startswith("SOFT") or note.startswith("HARD"))
        # Per spec: only audit PRIMARY entries; DIVERSITY/SOFT/HARD/DEPRECATED untouched
        tier = "PRIMARY" if is_primary_active else (
            "DEPRECATED" if note.startswith("DEPRECATED") else
            "DIVERSITY" if note.startswith("DIVERSITY") else
            "SOFT" if note.startswith("SOFT") else
            "HARD" if note.startswith("HARD") else "OTHER"
        )
        if tier != "PRIMARY":
            continue

        path = SUBMISSIONS / fn
        if not path.exists():
            rows.append(dict(file=fn, tier=tier, te_rae=np.nan,
                             pred_oof_rae=np.nan, delta=np.nan,
                             status="MISSING_FILE", note=note))
            continue

        df = pd.read_csv(path)
        pred_col = None
        for c in df.columns:
            if c.lower() in ("pec50", "predicted_pec50", "ypred", "pred"):
                pred_col = c; break
        if pred_col is None:
            rows.append(dict(file=fn, tier=tier, te_rae=np.nan,
                             pred_oof_rae=np.nan, delta=np.nan,
                             status="NO_PEC50_COL", note=note))
            continue
        pred_full = df[pred_col].to_numpy(dtype=float)
        if pred_full.shape[0] != 513:
            rows.append(dict(file=fn, tier=tier, te_rae=np.nan,
                             pred_oof_rae=np.nan, delta=np.nan,
                             status=f"WRONG_LEN_{pred_full.shape[0]}", note=note))
            continue
        pred_unb = pred_full[UNB_IDX]
        te_rae   = rae(UNB_Y, pred_unb)

        nb = nb_id(fn)
        pred_oof_rae = np.nan
        pred_oof_path = DATA_PROCESSED / f"{nb}_pred_oof.npy"
        if pred_oof_path.exists():
            try:
                p = np.load(pred_oof_path)
                if p.shape[0] == 513:
                    pred_oof_rae = rae(UNB_Y, p[UNB_IDX])
                elif p.shape[0] == 253:
                    pred_oof_rae = rae(UNB_Y, p)
                else:
                    pred_oof_rae = np.nan
            except Exception:
                pred_oof_rae = np.nan
        if np.isnan(pred_oof_rae):
            # Fall back to claimed RAE from the LADDER note
            cl = parse_claimed_rae(note)
            if cl is not None: pred_oof_rae = cl

        delta = abs(te_rae - pred_oof_rae) if not np.isnan(pred_oof_rae) else np.nan
        if np.isnan(delta):
            status = "NO_REF"
        elif delta > TOL:
            status = "FAIL"
        else:
            status = "OK"
        rows.append(dict(file=fn, tier=tier, te_rae=te_rae,
                         pred_oof_rae=pred_oof_rae, delta=delta,
                         status=status, note=note))

    return pd.DataFrame(rows)

def demote_failures(audit_df: pd.DataFrame):
    failures = set(audit_df.loc[audit_df["status"] == "FAIL", "file"].tolist())
    if not failures:
        print("\nNo PRIMARY demotions required.")
        return 0
    src = LADDER_PATH.read_text(encoding="utf-8")
    n_demoted = 0
    for fn in failures:
        # Find line containing "<fn>" and rewrite its note to start with DEPRECATED-AUDIT
        lines = src.splitlines()
        for i, ln in enumerate(lines):
            if f'"{fn}"' in ln and "PRIMARY" in ln:
                # Replace "PRIMARY-X:" prefix with "DEPRECATED-AUDIT:"
                new_ln = re.sub(r'"(PRIMARY-\d+:[^"]*)"',
                                lambda m: '"DEPRECATED-AUDIT: ' + m.group(1).split(":", 1)[1].strip()
                                          + ' -- te_RAE deviation >0.05 vs pred_oof"',
                                ln)
                if new_ln != ln:
                    lines[i] = new_ln
                    n_demoted += 1
                    break
        src = "\n".join(lines)
    LADDER_PATH.write_text(src + ("\n" if not src.endswith("\n") else ""), encoding="utf-8")
    print(f"\nDemoted {n_demoted} PRIMARY entries to DEPRECATED-AUDIT.")
    return n_demoted

def main():
    df = audit()
    print("\n=== PRIMARY-tier integrity audit (n={}) ===".format(len(df)))
    print(df[["file", "te_rae", "pred_oof_rae", "delta", "status"]].to_string(index=False))

    n_demoted = demote_failures(df)

    # Reload LADDER (after rewrite) to print final PRIMARY top-5
    import importlib, auto_submit_ladder
    importlib.reload(auto_submit_ladder)
    from auto_submit_ladder import LADDER as L2  # type: ignore

    primaries = []
    for fn, note in L2:
        if note.startswith("PRIMARY"):
            row = df[df["file"] == fn]
            if len(row) == 1:
                primaries.append((fn, float(row["te_rae"].iloc[0]),
                                  float(row["pred_oof_rae"].iloc[0])
                                  if not np.isnan(row["pred_oof_rae"].iloc[0]) else np.nan,
                                  row["status"].iloc[0]))
    primaries.sort(key=lambda x: (x[1] if not np.isnan(x[1]) else 1e9))
    print("\n=== FINAL PRIMARY top-5 (post-demotion, sorted by te_RAE on 253 unblind) ===")
    for rk, (fn, te, oof, st) in enumerate(primaries[:5], 1):
        oof_s = f"{oof:.4f}" if not np.isnan(oof) else "  n/a"
        print(f"  #{rk}  te_RAE={te:.4f}  pred_oof_RAE={oof_s}  status={st}  {fn}")
    return n_demoted, primaries

if __name__ == "__main__":
    main()

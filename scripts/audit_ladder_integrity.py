"""audit_ladder_integrity.py -- Pre-submission integrity audit.

Per feedback_te_vs_pred_oof_protocol.md (2026-06-08 update):
  - `te_X.npy` on 513 = deploy refit (model trained on ALL labels incl. the 253
    unblind). `te[unb_idx]` is therefore an IN-SAMPLE evaluation -- the deploy
    refit has SEEN those labels at fit time.
  - `X_pred_oof.npy` on 253 = honest 5-fold cross-fit (LB-faithful).
  - A gap of te[unb]_RAE << pred_oof_RAE is EXPECTED in-sample optimism
    (typically +0.20-0.40 RAE), NOT a contamination signal.
  - REAL contamination signal:
      (a) sha256(pred[unb_idx]) == sha256(_audit_unblind_y) -- bit-identical, OR
      (b) Pearson(pred[unb_idx], truth) >= 0.999  AND  te_RAE < 0.30
          (near-perfect in-sample fit, consistent with training leak).

For every PRIMARY / DEPRECATED-AUDIT entry in auto_submit_ladder.LADDER:
  1. Load submissions/<file> -> pec50 column
  2. Compute te_RAE / Pearson / sha256(pred[unb_idx]) on the 253 unblind labels
     (_audit_unblind_idx.npy / _audit_unblind_y.npy)
  3. Compare to the stored pred_oof RAE (data/processed/<nbXXX>_pred_oof.npy)
     - INFORMATIONAL only -- not used for demotion.
  4. DEMOTE (PRIMARY -> DEPRECATED-AUDIT) ONLY if rule (a) or (b) above fires.
  5. RESTORE (DEPRECATED-AUDIT -> PRIMARY-RESTORED) entries that no longer
     match the contamination criteria (e.g. demoted by an older te-gap rule).
  6. Print final PRIMARY top-5 ranked by te_RAE_on_unblind.

Outputs:
  - prints audit table to stdout
  - rewrites scripts/auto_submit_ladder.py with the demotion/restoration deltas
"""
import os, sys, re, hashlib
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

# Contamination thresholds (per feedback_te_vs_pred_oof_protocol)
PEARSON_LEAK_THRESHOLD = 0.999    # near-perfect correlation to truth
INSAMPLE_LEAK_RAE_MAX  = 0.30     # combined with high Pearson -> suspicious fit
# Informational threshold (NOT demotion): expected in-sample optimism gap
INFO_INSAMPLE_GAP_MIN  = 0.10     # te[unb]_RAE < pred_oof - 0.10 -> "expected"


def truth_sha256() -> str:
    return hashlib.sha256(np.asarray(UNB_Y, dtype=float).tobytes()).hexdigest()


def arr_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype=float).tobytes()).hexdigest()


def rae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - y_true.mean()))
    return float(num / den) if den > 0 else float("nan")


def safe_pearson(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def parse_claimed_rae(note: str):
    m = re.search(r"RAE\s+(0\.\d{3,4})", note)
    if m: return float(m.group(1))
    m = re.search(r"(0\.\d{3,4})\b", note)
    if m: return float(m.group(1))
    return None


def nb_id(filename: str):
    m = re.match(r"(nb\d+)", filename)
    return m.group(1) if m else None


def _is_primary_or_audit_demoted(note: str) -> bool:
    return note.startswith("PRIMARY") or note.startswith("DEPRECATED-AUDIT")


TRUTH_SHA = truth_sha256()


def audit():
    rows = []
    for fn, note in LADDER:
        # Audit PRIMARY entries (current candidates) AND DEPRECATED-AUDIT
        # entries (so the new rule can RESTORE them if they no longer meet the
        # contamination criteria). DIVERSITY / SOFT / HARD / legacy DEPRECATED
        # tags are not touched.
        if not _is_primary_or_audit_demoted(note):
            continue
        tier = "DEPRECATED-AUDIT" if note.startswith("DEPRECATED-AUDIT") else "PRIMARY"

        path = SUBMISSIONS / fn
        if not path.exists():
            rows.append(dict(file=fn, tier=tier, te_rae=np.nan,
                             pred_oof_rae=np.nan, pearson=np.nan,
                             sha256_leak=False, status="MISSING_FILE", note=note))
            continue

        df = pd.read_csv(path)
        pred_col = None
        for c in df.columns:
            if c.lower() in ("pec50", "predicted_pec50", "ypred", "pred"):
                pred_col = c; break
        if pred_col is None:
            rows.append(dict(file=fn, tier=tier, te_rae=np.nan,
                             pred_oof_rae=np.nan, pearson=np.nan,
                             sha256_leak=False, status="NO_PEC50_COL", note=note))
            continue
        pred_full = df[pred_col].to_numpy(dtype=float)
        if pred_full.shape[0] != 513:
            rows.append(dict(file=fn, tier=tier, te_rae=np.nan,
                             pred_oof_rae=np.nan, pearson=np.nan,
                             sha256_leak=False,
                             status=f"WRONG_LEN_{pred_full.shape[0]}", note=note))
            continue

        pred_unb  = pred_full[UNB_IDX]
        te_rae_v  = rae(UNB_Y, pred_unb)
        pearson_v = safe_pearson(pred_unb, UNB_Y)
        sha_leak  = (arr_sha256(pred_unb) == TRUTH_SHA)

        # pred_oof reference (honest cross-fit) -- informational only
        nb = nb_id(fn)
        pred_oof_rae = np.nan
        if nb is not None:
            pred_oof_path = DATA_PROCESSED / f"{nb}_pred_oof.npy"
            if pred_oof_path.exists():
                try:
                    p = np.load(pred_oof_path)
                    if p.shape[0] == 513:
                        pred_oof_rae = rae(UNB_Y, p[UNB_IDX])
                    elif p.shape[0] == 253:
                        pred_oof_rae = rae(UNB_Y, p)
                except Exception:
                    pass
        if np.isnan(pred_oof_rae):
            cl = parse_claimed_rae(note)
            if cl is not None: pred_oof_rae = cl

        # === NEW DEMOTION RULE ===
        # Demote ONLY for true contamination indicators on the deploy CSV.
        # Expected in-sample optimism (te[unb] << pred_oof) is NOT demoted.
        if sha_leak:
            status = "FAIL_SHA256_LEAK"
        elif (not np.isnan(pearson_v)) and pearson_v >= PEARSON_LEAK_THRESHOLD \
                and (not np.isnan(te_rae_v)) and te_rae_v < INSAMPLE_LEAK_RAE_MAX:
            status = "FAIL_NEAR_PERFECT_FIT"
        else:
            # Not contamination. Annotate expected in-sample optimism if present.
            if (not np.isnan(pred_oof_rae)) and (not np.isnan(te_rae_v)) \
                    and (pred_oof_rae - te_rae_v) > INFO_INSAMPLE_GAP_MIN:
                status = "OK_INSAMPLE_GAP_EXPECTED"
            else:
                status = "OK"

        rows.append(dict(file=fn, tier=tier, te_rae=te_rae_v,
                         pred_oof_rae=pred_oof_rae, pearson=pearson_v,
                         sha256_leak=sha_leak, status=status, note=note))

    return pd.DataFrame(rows)


def _restore_primary_tag(line: str) -> str:
    """Convert a 'DEPRECATED-AUDIT: ... -- te_RAE deviation >0.05 vs pred_oof'
    line back to a PRIMARY-RESTORED tag (so the auto-submitter re-includes it).
    Preserves the original body of the note minus the audit tail."""
    # Strip the canonical te_RAE-deviation tail produced by the old rule
    line2 = re.sub(r'\s*--\s*te_RAE deviation >0\.05 vs pred_oof"',
                   '"', line)
    return re.sub(r'"DEPRECATED-AUDIT:\s*', '"PRIMARY-RESTORED: ', line2)


def reconcile_ladder(audit_df: pd.DataFrame):
    """Apply the NEW rule against the current ladder text:
       - Demote (PRIMARY -> DEPRECATED-AUDIT) entries flagged FAIL_*
       - Restore (DEPRECATED-AUDIT -> PRIMARY-RESTORED) entries that are OK*
       Returns (n_demoted, n_restored, demoted_files, restored_files).
    """
    src = LADDER_PATH.read_text(encoding="utf-8")
    lines = src.splitlines()
    n_demoted = 0
    n_restored = 0
    demoted_files = []
    restored_files = []

    fail_files = set(audit_df.loc[audit_df["status"].str.startswith("FAIL"), "file"].tolist())
    restore_files = set(audit_df.loc[
        (audit_df["tier"] == "DEPRECATED-AUDIT") & (audit_df["status"].str.startswith("OK")),
        "file",
    ].tolist())

    for i, ln in enumerate(lines):
        # Demotion: PRIMARY -> DEPRECATED-AUDIT (true contamination only)
        for fn in fail_files:
            if f'"{fn}"' in ln and "PRIMARY-" in ln and "DEPRECATED" not in ln:
                new_ln = re.sub(
                    r'"(PRIMARY-\d+:[^"]*)"',
                    lambda m: '"DEPRECATED-AUDIT: ' + m.group(1).split(":", 1)[1].strip()
                              + ' -- TRUE_CONTAMINATION (sha256/near-perfect-fit)"',
                    ln,
                )
                if new_ln != ln:
                    lines[i] = new_ln
                    n_demoted += 1
                    demoted_files.append(fn)
                    break
        # Restoration: DEPRECATED-AUDIT -> PRIMARY-RESTORED (no longer leaking)
        for fn in restore_files:
            if f'"{fn}"' in ln and "DEPRECATED-AUDIT" in ln:
                new_ln = _restore_primary_tag(ln)
                if new_ln != ln:
                    lines[i] = new_ln
                    n_restored += 1
                    restored_files.append(fn)
                    break

    if n_demoted == 0 and n_restored == 0:
        print("\nNo ladder changes required.")
        return 0, 0, [], []
    new_src = "\n".join(lines)
    LADDER_PATH.write_text(new_src + ("\n" if not new_src.endswith("\n") else ""),
                           encoding="utf-8")
    print(f"\nLadder updated: demoted={n_demoted}, restored={n_restored}.")
    if demoted_files:
        print("  Demoted (TRUE_CONTAMINATION):")
        for fn in demoted_files: print(f"    - {fn}")
    if restored_files:
        print("  Restored (PRIMARY-RESTORED, no contamination):")
        for fn in restored_files: print(f"    - {fn}")
    return n_demoted, n_restored, demoted_files, restored_files


def main():
    df = audit()
    print("\n=== Ladder integrity audit (PRIMARY + DEPRECATED-AUDIT, n={}) ===".format(len(df)))
    show_cols = ["file", "tier", "te_rae", "pred_oof_rae", "pearson",
                 "sha256_leak", "status"]
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df[show_cols].to_string(index=False))

    n_demoted, n_restored, _, _ = reconcile_ladder(df)

    # Reload LADDER (after rewrite) to print final PRIMARY top-5
    import importlib, auto_submit_ladder
    importlib.reload(auto_submit_ladder)
    from auto_submit_ladder import LADDER as L2  # type: ignore

    primaries = []
    for fn, note in L2:
        if note.startswith("PRIMARY"):
            row = df[df["file"] == fn]
            if len(row) == 1:
                primaries.append((fn,
                                  float(row["te_rae"].iloc[0]),
                                  float(row["pred_oof_rae"].iloc[0])
                                  if not np.isnan(row["pred_oof_rae"].iloc[0]) else np.nan,
                                  row["status"].iloc[0]))
    primaries.sort(key=lambda x: (x[1] if not np.isnan(x[1]) else 1e9))
    print("\n=== FINAL PRIMARY top-5 (post-reconcile, sorted by te_RAE on 253 unblind) ===")
    for rk, (fn, te, oof, st) in enumerate(primaries[:5], 1):
        oof_s = f"{oof:.4f}" if not np.isnan(oof) else "  n/a"
        print(f"  #{rk}  te_RAE={te:.4f}  pred_oof_RAE={oof_s}  status={st}  {fn}")
    return n_demoted, n_restored, primaries


if __name__ == "__main__":
    main()

"""Integrity audit for te_*.npy: detect truth-contaminated test predictions."""
import os
import glob
import numpy as np
import pandas as pd

PROC = r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/processed"
RAW = r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/raw"


def rae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.mean(np.abs(y_true - y_pred))
    den = np.mean(np.abs(y_true - np.mean(y_true)))
    return float(num / den) if den > 0 else float("nan")


def load_unblind():
    yp = os.path.join(PROC, "_audit_unblind_y.npy")
    ip = os.path.join(PROC, "_audit_unblind_idx.npy")
    if os.path.exists(yp) and os.path.exists(ip):
        y = np.load(yp)
        idx = np.load(ip)
        return y, idx
    # Recompute
    blind = pd.read_csv(os.path.join(RAW, "pxr-challenge_TEST_BLINDED.csv"))
    un = pd.read_csv(os.path.join(RAW, "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"))
    # determine join key & label col
    bcol = "Molecule Name" if "Molecule Name" in blind.columns else blind.columns[0]
    ucol = "Molecule Name" if "Molecule Name" in un.columns else un.columns[0]
    # label
    label_col = None
    for c in ["pEC50", "pec50", "PEC50"]:
        if c in un.columns:
            label_col = c
            break
    if label_col is None:
        # try any column with pec50
        for c in un.columns:
            if "pec50" in c.lower():
                label_col = c
                break
    pos = {n: i for i, n in enumerate(blind[bcol].astype(str).tolist())}
    idx_list, y_list = [], []
    for _, row in un.iterrows():
        name = str(row[ucol])
        if name in pos and pd.notna(row[label_col]):
            idx_list.append(pos[name])
            y_list.append(float(row[label_col]))
    idx = np.array(idx_list, dtype=int)
    y = np.array(y_list, dtype=float)
    np.save(yp, y)
    np.save(ip, idx)
    return y, idx


def main():
    y_un, idx_un = load_unblind()
    print(f"Unblind: n={len(y_un)}, idx range [{idx_un.min()},{idx_un.max()}]")

    te_files = sorted(glob.glob(os.path.join(PROC, "te_*.npy")))
    print(f"Found {len(te_files)} te_*.npy files")

    rows = []
    for tf in te_files:
        name = os.path.basename(tf).replace(".npy", "")
        try:
            arr = np.load(tf)
        except Exception as e:
            rows.append({"file": name, "te_rae": np.nan, "pred_oof_rae": np.nan,
                         "gap": np.nan, "flag": f"load_error:{e}"})
            continue
        if arr.shape[0] != 513:
            rows.append({"file": name, "te_rae": np.nan, "pred_oof_rae": np.nan,
                         "gap": np.nan, "flag": f"bad_shape:{arr.shape}"})
            continue
        if arr.ndim > 1:
            # feature array, not a prediction — skip
            rows.append({"file": name, "te_rae": np.nan, "pred_oof_rae": np.nan,
                         "gap": np.nan, "flag": f"feature_array:{arr.shape}"})
            continue
        te_rae = rae(y_un, arr[idx_un])

        # find matching pred_oof
        stem = name[len("te_"):]
        po_path = os.path.join(PROC, f"{stem}_pred_oof.npy")
        pred_oof_rae = np.nan
        gap = np.nan
        flag = "ok"
        if os.path.exists(po_path):
            try:
                po = np.load(po_path)
                if po.shape[0] == len(y_un):
                    pred_oof_rae = rae(y_un, po)
                    gap = pred_oof_rae - te_rae
                    if gap > 0.05:
                        flag = "CONTAMINATED"
                    elif gap > 0.02:
                        flag = "suspicious"
                elif po.shape[0] == 513:
                    pred_oof_rae = rae(y_un, po[idx_un])
                    gap = pred_oof_rae - te_rae
                    if gap > 0.05:
                        flag = "CONTAMINATED"
                    elif gap > 0.02:
                        flag = "suspicious"
                else:
                    flag = f"oof_bad_shape:{po.shape}"
            except Exception as e:
                flag = f"oof_load_error:{e}"
        else:
            # heuristic: if te_rae is implausibly low (< 0.30) flag it
            if te_rae < 0.30:
                flag = "SUSPECT_LOW_NO_OOF"
        rows.append({"file": name, "te_rae": te_rae,
                     "pred_oof_rae": pred_oof_rae, "gap": gap, "flag": flag})

    df = pd.DataFrame(rows).sort_values("te_rae")
    out = os.path.join(PROC, "integrity_audit.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")

    contaminated = df[df["flag"] == "CONTAMINATED"]
    suspect_low = df[df["flag"] == "SUSPECT_LOW_NO_OOF"]
    print(f"\nCONTAMINATED (gap>0.05): {len(contaminated)}")
    for _, r in contaminated.iterrows():
        print(f"  {r['file']}: te={r['te_rae']:.4f} oof={r['pred_oof_rae']:.4f} gap={r['gap']:.4f}")
    print(f"\nSUSPECT (te_rae<0.30, no oof): {len(suspect_low)}")
    for _, r in suspect_low.iterrows():
        print(f"  {r['file']}: te={r['te_rae']:.4f}")

    # explicit suspect file check
    suspects = ["te_nb562", "te_nb700", "te_nb701", "te_nb713", "te_nb703"]
    print("\n--- SUSPECT FILES ---")
    for s in suspects:
        row = df[df["file"] == s]
        if len(row) == 0:
            print(f"  {s}: NOT FOUND")
        else:
            r = row.iloc[0]
            print(f"  {s}: te={r['te_rae']:.4f} oof={r['pred_oof_rae']} gap={r['gap']} flag={r['flag']}")


if __name__ == "__main__":
    main()

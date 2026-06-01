"""nb620 -- HONEST APPLES-TO-APPLES RE-EVALUATION of nb610-614 vs nb562.

The previous comparison (nb610 cross-fit RAE 0.4277 etc.) used te_nb562.npy as
the anchor — but te_nb562.npy is the DEPLOY refit (rank-stretch s=1.10 fit on
all 253 unblind labels via SLSQP/post-hoc), which is IN-SAMPLE on those 253.
So that "cross-fit" was a candidate's honest residual stacked on top of an
in-sample anchor.

The HONEST cross-fit comparison for the anchor itself is nb562_pred_oof.npy
(per-fold cross-fit s=1.10) -> RAE 0.5065.

For each nb610-614 candidate the right honest-vs-honest comparisons are:

  (A) candidate's OWN-anchor honest cross-fit pred_oof  (already saved on disk)
      = anchor[unb_idx] + alpha[unb_idx] * resid_oof
      where `anchor` is whatever anchor the candidate was built on.
      This is the candidate's true honest score.

  (B) nb562-anchored fair cross-fit
      = nb562_pred_oof + alpha[unb_idx] * resid_oof
      This swaps the candidate's IN-SAMPLE anchor for nb562's HONEST cross-fit
      anchor, holding the cross-fit residual the same. It is the
      apples-to-apples test of "does this router's residual ADD value on top
      of an honest nb562?". Note: resid_oof was fit to truth - candidate_anchor,
      so this is structurally biased against the candidate when its anchor is
      not nb562; the bias makes the test conservative (a candidate that wins
      here truly carries orthogonal signal).

The baseline to beat: nb562_pred_oof honest RAE = 0.5065.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW

TAG = "nb620"
BASELINE_RAE = 0.5065

# Each candidate: original anchor (used in-sample to build te_*.npy and to
# define the residual target) + cross-fit artefacts saved on disk.
CANDIDATES = [
    {
        "tag": "nb610",
        "anchor_name": "nb562",
        "anchor_file": "te_nb562.npy",     # in-sample deploy refit
        "resid_oof":   "nb610_resid_oof.npy",
        "alpha":       "te_nb610_alpha.npy",
        "pred_oof":    "nb610_pred_oof.npy",
    },
    {
        "tag": "nb611",
        "anchor_name": "nb503",
        "anchor_file": "te_nb503.npy",
        "resid_oof":   "nb611_resid_oof.npy",
        "alpha":       "te_nb611_alpha.npy",
        "pred_oof":    "nb611_pred_oof.npy",
    },
    {
        "tag": "nb612",
        "anchor_name": "chemprop_aux",
        "anchor_file": "te_chemprop_aux.npy",
        "resid_oof":   "nb612_resid_oof.npy",
        "alpha":       "te_nb612_alpha.npy",
        "pred_oof":    "nb612_pred_oof.npy",
    },
    {
        "tag": "nb613",
        "anchor_name": "nb464",
        "anchor_file": "te_nb464.npy",
        "resid_oof":   "nb613_resid_oof.npy",
        "alpha":       "te_nb613_alpha.npy",
        "pred_oof":    None,  # nb613 did NOT save pred_oof; reconstruct
    },
    {
        "tag": "nb614",
        "anchor_name": "blend(nb562+601+610+611+612+613)",
        "anchor_file": None,                # nb614 is itself a blend
        "resid_oof":   None,
        "alpha":       None,
        "pred_oof":    "nb614_pred_oof.npy",  # SLSQP cross-fit OOF (already honest)
    },
]


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- HONEST cross-fit re-eval of nb610-614 vs nb562")
    print("=" * 78)

    needed = {
        "TEST_BLINDED":   DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":      DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb562_pred_oof": DATA_PROCESSED / "nb562_pred_oof.npy",
        "te_nb562":       DATA_PROCESSED / "te_nb562.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING base inputs:", missing)
        return {"success": False, "notes": "missing_inputs", "missing": missing}

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"].tolist())}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_unb = len(unb_idx)
    print(f"unblind n={n_unb}")

    nb562_oof = np.load(needed["nb562_pred_oof"]).astype(np.float64)
    nb562_te  = np.load(needed["te_nb562"]).astype(np.float64)
    rae_562_honest   = float(rae(unb_y, nb562_oof))
    rae_562_insample = float(rae(unb_y, nb562_te[unb_idx]))
    print(f"nb562 HONEST cross-fit RAE = {rae_562_honest:.4f}   "
          f"(claimed 0.5065)")
    print(f"nb562 IN-SAMPLE deploy RAE = {rae_562_insample:.4f}  "
          f"(this is what nb610-613 wrongly stacked on)")

    rows = []
    for cfg in CANDIDATES:
        tag = cfg["tag"]
        print("\n" + "-" * 78)
        print(f"{tag}  anchor={cfg['anchor_name']}")

        # --- (A) candidate's OWN honest cross-fit (its own anchor + resid) ---
        own_honest_rae = None
        own_anchor_insample_rae = None
        if cfg["pred_oof"] is not None:
            p = DATA_PROCESSED / cfg["pred_oof"]
            if p.exists():
                pred_oof_own = np.load(p).astype(np.float64)
                if pred_oof_own.shape != (n_unb,):
                    print(f"  ! pred_oof shape {pred_oof_own.shape} != ({n_unb},)")
                else:
                    own_honest_rae = float(rae(unb_y, pred_oof_own))
                    print(f"  (A) own-anchor honest cross-fit RAE = {own_honest_rae:.4f}")
        elif tag == "nb613":
            # Reconstruct nb613 honest OOF from its anchor (nb464) + resid + alpha
            te_anchor_464 = np.load(DATA_PROCESSED / "te_nb464.npy").astype(np.float64)
            alpha_613 = np.load(DATA_PROCESSED / "te_nb613_alpha.npy").astype(np.float64)
            resid_613 = np.load(DATA_PROCESSED / "nb613_resid_oof.npy").astype(np.float64)
            # IMPORTANT: te_nb464.npy is itself an in-sample refit (deploy) for nb464,
            # so this is NOT pure honest cross-fit; it is the same kind of artefact
            # nb613 used internally. nb464 had no public pred_oof saved.
            pred_oof_own = te_anchor_464[unb_idx] + alpha_613[unb_idx] * resid_613
            own_honest_rae = float(rae(unb_y, pred_oof_own))
            print(f"  (A) own-anchor cross-fit RAE = {own_honest_rae:.4f}  "
                  f"(NOTE: nb464 anchor itself is in-sample deploy; no pred_oof on disk)")

        # In-sample anchor RAE (what nb610-613 actually stacked on)
        if cfg["anchor_file"] is not None:
            af = DATA_PROCESSED / cfg["anchor_file"]
            if af.exists():
                anc = np.load(af).astype(np.float64)
                own_anchor_insample_rae = float(rae(unb_y, anc[unb_idx]))
                print(f"      anchor te_{cfg['anchor_name']} IN-SAMPLE RAE = "
                      f"{own_anchor_insample_rae:.4f}")

        # --- (B) Fair: swap to nb562 HONEST cross-fit anchor + this resid ---
        fair_nb562_rae = None
        if cfg["resid_oof"] is not None and cfg["alpha"] is not None:
            resid = np.load(DATA_PROCESSED / cfg["resid_oof"]).astype(np.float64)
            alpha = np.load(DATA_PROCESSED / cfg["alpha"]).astype(np.float64)
            if resid.shape != (n_unb,):
                print(f"  ! resid shape {resid.shape} != ({n_unb},)")
            else:
                fair_pred = nb562_oof + alpha[unb_idx] * resid
                fair_nb562_rae = float(rae(unb_y, fair_pred))
                print(f"  (B) nb562_oof + alpha*resid   RAE = {fair_nb562_rae:.4f}  "
                      f"(apples-to-apples vs 0.5065)")
        elif tag == "nb614":
            # nb614 is itself a blended SLSQP cross-fit OOF, no resid to graft.
            # Its pred_oof IS already an honest cross-fit (5-fold SLSQP).
            print("  (B) n/a -- nb614 is a blend; its pred_oof IS the honest score")

        delta_vs_562 = (
            (fair_nb562_rae - rae_562_honest)
            if fair_nb562_rae is not None
            else None
        )
        delta_own = (
            (own_honest_rae - rae_562_honest)
            if own_honest_rae is not None
            else None
        )

        rows.append({
            "tag": tag,
            "anchor": cfg["anchor_name"],
            "own_honest_rae": own_honest_rae,
            "anchor_insample_rae": own_anchor_insample_rae,
            "fair_nb562_anchored_rae": fair_nb562_rae,
            "nb562_honest_rae": rae_562_honest,
            "delta_own_vs_nb562_honest": delta_own,
            "delta_fair_vs_nb562_honest": delta_vs_562,
            "beats_nb562_in_own_honest": (
                bool(own_honest_rae is not None and own_honest_rae < rae_562_honest)
            ),
            "beats_nb562_in_fair": (
                bool(fair_nb562_rae is not None and fair_nb562_rae < rae_562_honest)
            ),
        })

    # ---------- Summary table ----------
    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY -- HONEST vs HONEST ===")
    print(f"  nb562 baseline (HONEST cross-fit) RAE = {rae_562_honest:.4f}\n")
    print(f"  {'tag':<6}  {'anchor':<32}  "
          f"{'own_honest':>10}  {'fair_562':>9}  "
          f"{'d_own':>7}  {'d_fair':>7}  beats_fair")
    print("  " + "-" * 100)
    for r in rows:
        own = (f"{r['own_honest_rae']:.4f}"
               if r['own_honest_rae'] is not None else "  n/a ")
        fair = (f"{r['fair_nb562_anchored_rae']:.4f}"
                if r['fair_nb562_anchored_rae'] is not None else "  n/a ")
        d_own = (f"{r['delta_own_vs_nb562_honest']:+.4f}"
                 if r['delta_own_vs_nb562_honest'] is not None else "   n/a")
        d_fair = (f"{r['delta_fair_vs_nb562_honest']:+.4f}"
                  if r['delta_fair_vs_nb562_honest'] is not None else "   n/a")
        beats = "YES" if r["beats_nb562_in_fair"] else "no"
        print(f"  {r['tag']:<6}  {r['anchor']:<32}  "
              f"{own:>10}  {fair:>9}  {d_own:>7}  {d_fair:>7}  {beats}")

    truly_beat = [
        r["tag"] for r in rows if r["beats_nb562_in_fair"]
    ]
    demote = [
        r["tag"] for r in rows
        if not r["beats_nb562_in_fair"]
        and r["fair_nb562_anchored_rae"] is not None
    ]
    print(f"\n  candidates that beat nb562 in FAIR honest cmp: {truly_beat}")
    print(f"  candidates to DEMOTE from PRIMARY:             {demote}")
    print("=" * 78)

    # Persist a tidy CSV next to processed dir
    out_csv = DATA_PROCESSED / f"{TAG}_honest_reeval.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    return {
        "success": True,
        "nb562_honest_rae": rae_562_honest,
        "nb562_insample_rae": rae_562_insample,
        "rows": rows,
        "truly_beat_nb562": truly_beat,
        "demote_from_primary": demote,
        "out_csv": str(out_csv),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        if k == "rows":
            continue
        print(f"  {k}: {v}")

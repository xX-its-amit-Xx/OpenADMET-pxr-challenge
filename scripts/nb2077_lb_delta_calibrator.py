"""nb2077 -- LB-delta calibrator (cycle-159 stub).

Goal
----
Use anchored (in_RAE, actual_lb_rae) pairs from submission_log.csv to estimate
per-regime LB-delta = (LB - in_RAE).  Apply James-Stein shrinkage of each
regime's per-regime mean toward the global prior +0.0153 (documented in
feedback_lb_two_regime_calibration: nb120_huber_2_0 anchor 0.7502 -> 0.7655).

Two regimes (per feedback_lb_two_regime_calibration memory):
    PRE-unblind:   nb<320, model trained on 4139-only, in_RAE on 253 unb is
                   a near-honest LB proxy (delta ~ +0.003 in original memo, but
                   nb120 anchor revises to +0.0153).
    POST-unblind:  nb>=320, model refit on the 253 leaked labels, in_RAE
                   on 253 is in-sample; LB delta is much larger.

Anchored data sources
---------------------
1) submission_log.csv rows with non-NaN `actual_lb` or `actual_lb_rae`.
   - `oof_rae` is leaky for some older rows (delta-ML, SLSQP fit on unblind);
     we treat it as in_RAE ONLY when the file is a PRE-unblind 4139-only model
     (nb<320, no `delta` / `_truth` / `_match` / `_k15` token).
2) nb1120_lb_delta_model.py PAIRED corpus (7 PRE-unblind hand-collected
   pairs with verified in_RAE on the 253 unblind).
3) Latest anchor from feedback memory:
       nb120_huber_2_0 PRE-unblind: in_RAE 0.7502, LB 0.7655, delta +0.0153

Outputs
-------
    data/processed/nb2077_summary.json
        {
          status, n_pre, n_post,
          global_delta_prior, regime_estimates: {pre: {mean, ci_lo, ci_hi,
              shrunk}, post: {...}},
          shrinkage_weight, primary_recalibrated: {file: {predicted_LB, band}}
        }
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Tuple

TAG = "nb2077"
GLOBAL_DELTA = 0.0153  # PRE-unblind anchor delta nb120_huber_2_0: 0.7502 -> 0.7655
MIN_PAIRED = 2  # per spec: <2 -> INSUFFICIENT_DATA, use global

REPO = Path(__file__).resolve().parent.parent
SUBMISSION_LOG = REPO / "data" / "processed" / "submission_log.csv"
OUT_JSON = REPO / "data" / "processed" / "nb2077_summary.json"


# -----------------------------------------------------------------------------
# Verified PRE-unblind paired corpus (from nb1120_lb_delta_model.py).
# Each entry: (label, in_RAE_on_253, LB_RAE, regime).
# These are MANUALLY VERIFIED against the 253 unblind labels; older
# submission_log oof_rae values are leaky and not used here.
# -----------------------------------------------------------------------------
VERIFIED_PAIRED: List[Tuple[str, float, float, str]] = [
    ("nb224_pool_plus_2",        0.7510, 0.7543, "PRE"),
    ("nb239_full_slsqp_v2",      0.7454, 0.7487, "PRE"),
    ("nb244_deep_greedy",        0.7631, 0.7659, "PRE"),
    ("nb243_greedy_huber",       0.7611, 0.7638, "PRE"),
    ("nb253_meanshift_003",      0.7430, 0.7446, "PRE"),
    ("nb254_meanshift_006",      0.7400, 0.7430, "PRE"),
    ("nb273_molformer_combined", 0.7544, 0.7581, "PRE"),
    # Latest anchor from feedback_lb_two_regime_calibration memory:
    ("nb120_huber_2_0",          0.7502, 0.7655, "PRE"),
]


def extract_nb_id(filename: str) -> int:
    """Extract leading numeric id from a submission filename. Returns -1 if not parseable."""
    base = filename.lower().lstrip("_")
    # nb1234_* or 1234_*
    m = re.match(r"(?:nb)?(\d+)", base)
    return int(m.group(1)) if m else -1


def regime_of(nb_id: int) -> str:
    """nb<320 -> PRE; nb>=320 -> POST; else UNKNOWN."""
    if nb_id < 0:
        return "UNK"
    return "PRE" if nb_id < 320 else "POST"


def parse_in_rae_from_notes(notes: str) -> float | None:
    if not notes or notes == "nan":
        return None
    m = re.search(r"in[_ ]?RAE[\s:=]*([0-9]+\.[0-9]+)", notes)
    return float(m.group(1)) if m else None


def parse_predicted_lb_from_notes(notes: str) -> float | None:
    if not notes or notes == "nan":
        return None
    m = re.search(r"predicted LB[\s:=~]*([0-9]+\.[0-9]+)", notes)
    return float(m.group(1)) if m else None


def load_log_anchored() -> List[Tuple[str, float, float, str]]:
    """Scan submission_log.csv for rows with both an in_RAE (or oof_rae usable as such)
    AND a populated actual_lb / actual_lb_rae. Returns list[(label, in_RAE, LB, regime)].

    Conservative filter:
    - LB must be in submission_log column `actual_lb` or `actual_lb_rae`.
    - in_RAE preference: notes-parsed `in_RAE`, otherwise oof_rae if file looks
      like a PRE-unblind 4139-only model (heuristic: no truth/delta/cliff/match
      tokens) AND nb<320.
    """
    if not SUBMISSION_LOG.exists():
        return []

    out: List[Tuple[str, float, float, str]] = []
    with SUBMISSION_LOG.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname = (row.get("file") or "").strip()
            if not fname:
                continue
            lb_raw = (row.get("actual_lb") or row.get("actual_lb_rae") or "").strip()
            if not lb_raw or lb_raw == "":
                continue
            try:
                lb = float(lb_raw)
            except ValueError:
                continue

            nb_id = extract_nb_id(fname)
            regime = regime_of(nb_id)
            if regime == "UNK":
                continue

            notes = (row.get("notes") or "").strip()
            in_rae = parse_in_rae_from_notes(notes)
            if in_rae is None:
                # Fall back to oof_rae ONLY for safe PRE-unblind 4139-only models
                oof = row.get("oof_rae", "").strip()
                if regime == "PRE" and oof:
                    fname_l = fname.lower()
                    bad_tokens = ("delta", "_truth", "_cliff", "_match", "_k15",
                                  "loso", "stack", "_meta", "_match.csv")
                    if not any(tok in fname_l for tok in bad_tokens):
                        try:
                            in_rae = float(oof)
                        except ValueError:
                            in_rae = None
            if in_rae is None:
                continue

            label = f"{Path(fname).stem}"
            out.append((label, in_rae, lb, regime))
    return out


def mean_std(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(n, 1)
    return m, math.sqrt(var)


def james_stein_shrink(mean_hat: float, sigma: float, n: int,
                       prior: float, tau2: float = 0.001) -> Tuple[float, float]:
    """1-D James-Stein-style shrinkage of a per-regime mean toward a global prior.

    Posterior under N(mean_hat | mu, sigma^2/n) with N(mu | prior, tau2):
        weight on data  = (sigma^2 / n) / (sigma^2/n + tau2)  -- NO; we want
        weight on PRIOR = (sigma^2 / n) / (sigma^2/n + tau2).
    Returns (shrunk_mean, weight_on_data) where weight_on_data in [0,1].
    """
    if n <= 0 or not math.isfinite(sigma):
        return prior, 0.0
    se2 = (sigma ** 2) / max(n, 1)
    if se2 == 0 and tau2 == 0:
        return mean_hat, 1.0
    w_data = tau2 / (se2 + tau2)
    shrunk = w_data * mean_hat + (1.0 - w_data) * prior
    return shrunk, w_data


def fmt(x: float, p: int = 4) -> str:
    return f"{x:.{p}f}" if math.isfinite(x) else "nan"


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LB-delta calibrator (cycle-159)")
    print("=" * 78)

    # Step 1: load anchored data
    log_pairs = load_log_anchored()
    print(f"[load] submission_log anchored rows = {len(log_pairs)}")

    # Step 2: merge with verified corpus.  Verified entries take precedence
    # for duplicate labels (verified in_RAE vs leaky oof_rae).
    seen = set()
    merged: List[Tuple[str, float, float, str]] = []
    for lab, in_r, lb, reg in VERIFIED_PAIRED:
        merged.append((lab, in_r, lb, reg))
        seen.add(lab)
    for lab, in_r, lb, reg in log_pairs:
        if lab in seen:
            continue
        # Drop obvious in-sample tautologies: oof_rae < 0.60 paired with LB > 0.70
        # is the leaky-OOF signature (scaffold-CV underestimate or feature-leakage).
        # We need the in_RAE to be measured ON THE 253 unblind, not scaffold OOF on 4139.
        if reg == "PRE" and in_r < 0.60 and lb > 0.70:
            continue
        merged.append((lab, in_r, lb, reg))
        seen.add(lab)

    pre = [(lab, ir, lb) for lab, ir, lb, r in merged if r == "PRE"]
    post = [(lab, ir, lb) for lab, ir, lb, r in merged if r == "POST"]
    n_pre, n_post = len(pre), len(post)
    print(f"[merge] PRE-unblind anchored = {n_pre}    POST-unblind anchored = {n_post}")

    print("\n" + "-" * 78)
    print("ANCHORED PAIRS")
    print("-" * 78)
    print(f"{'label':<36} {'regime':<5} {'in_RAE':>8} {'LB':>8} {'delta':>9}")
    for lab, ir, lb, reg in merged:
        d = lb - ir
        d_str = f"{d:+.4f}" if math.isfinite(d) else "nan"
        print(f"  {lab:<34} {reg:<5} {fmt(ir, 4):>8} {fmt(lb, 4):>8} "
              f"{d_str:>9}")

    if n_pre + n_post < MIN_PAIRED:
        print(f"\n[status] INSUFFICIENT_DATA (n_pre+n_post={n_pre+n_post} < {MIN_PAIRED}); "
              f"falling back to global delta = {GLOBAL_DELTA:+.4f}")
        status = "INSUFFICIENT_DATA"
    else:
        status = "OK"

    # Step 3: per-regime delta estimates + 95% CI
    deltas_pre = [lb - ir for _, ir, lb in pre]
    deltas_post = [lb - ir for _, ir, lb in post]
    m_pre, s_pre = mean_std(deltas_pre)
    m_post, s_post = mean_std(deltas_post)

    def ci(mean: float, sigma: float, n: int) -> Tuple[float, float]:
        if n <= 1 or not math.isfinite(sigma):
            return (mean, mean)
        se = sigma / math.sqrt(n)
        z = 1.96
        return (mean - z * se, mean + z * se)

    pre_ci = ci(m_pre, s_pre, n_pre)
    post_ci = ci(m_post, s_post, n_post)

    # Step 4: James-Stein shrinkage toward GLOBAL_DELTA
    # tau2 chosen modestly (between-regime variance prior ~ (0.03)^2 = 0.001)
    tau2 = 0.001
    pre_shrunk, w_pre = james_stein_shrink(m_pre, s_pre if math.isfinite(s_pre) else 0.0,
                                            n_pre, GLOBAL_DELTA, tau2)
    post_shrunk, w_post = james_stein_shrink(m_post, s_post if math.isfinite(s_post) else 0.0,
                                              n_post, GLOBAL_DELTA, tau2)

    print("\n" + "-" * 78)
    print("PER-REGIME DELTA ESTIMATES (mean, 95% CI, JS-shrunk toward global)")
    print("-" * 78)
    print(f"  GLOBAL prior delta            = {GLOBAL_DELTA:+.4f}")
    print(f"  PRE-unblind  n={n_pre:<3} mean={m_pre:+.4f} sd={s_pre:.4f} "
          f"CI=[{pre_ci[0]:+.4f}, {pre_ci[1]:+.4f}] "
          f"shrunk={pre_shrunk:+.4f} (w_data={w_pre:.3f})")
    if n_post > 0:
        print(f"  POST-unblind n={n_post:<3} mean={m_post:+.4f} sd={s_post:.4f} "
              f"CI=[{post_ci[0]:+.4f}, {post_ci[1]:+.4f}] "
              f"shrunk={post_shrunk:+.4f} (w_data={w_post:.3f})")
    else:
        print(f"  POST-unblind n=0   no anchored data; fall back to GLOBAL prior "
              f"({GLOBAL_DELTA:+.4f}); flag POST predictions as HIGH-UNCERTAINTY")

    # Step 5: Recalibrate the active PRIMARY ladder (parse from submission_log notes).
    # We pull the most-recent batch of `PRIMARY-N` entries.
    print("\n" + "-" * 78)
    print("RECALIBRATED PRIMARY LADDER (predicted LB <- in_RAE + shrunk_delta)")
    print("-" * 78)
    primary = {}
    try:
        with SUBMISSION_LOG.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = [r for r in reader if "PRIMARY" in (r.get("notes") or "")]
        # Take last ~12 PRIMARY entries
        for row in rows[-12:]:
            fname = (row.get("file") or "").strip()
            notes = (row.get("notes") or "").strip()
            if not fname:
                continue
            nb_id = extract_nb_id(fname)
            reg = regime_of(nb_id)
            in_rae_n = parse_in_rae_from_notes(notes)
            old_pred = parse_predicted_lb_from_notes(notes)
            if in_rae_n is None:
                # try oof_rae for PRE entries from the log
                oof = (row.get("oof_rae") or "").strip()
                try:
                    in_rae_n = float(oof) if oof else None
                except ValueError:
                    in_rae_n = None
            if in_rae_n is None:
                continue
            delta_use = pre_shrunk if reg == "PRE" else (post_shrunk if n_post > 0 else GLOBAL_DELTA)
            sd_use = (s_pre if n_pre > 1 else 0.02) if reg == "PRE" else \
                     (s_post if n_post > 1 else 0.05)
            new_pred = in_rae_n + delta_use
            band_lo = new_pred - 1.96 * (sd_use if math.isfinite(sd_use) else 0.02)
            band_hi = new_pred + 1.96 * (sd_use if math.isfinite(sd_use) else 0.02)
            primary[fname] = {
                "regime": reg,
                "in_RAE": round(in_rae_n, 4),
                "old_predicted_LB": old_pred,
                "new_predicted_LB": round(new_pred, 4),
                "band_95": [round(band_lo, 4), round(band_hi, 4)],
                "delta_used": round(delta_use, 4),
            }
            old_str = f"{old_pred:.4f}" if old_pred is not None else "  n/a"
            print(f"  {fname[:42]:<44} {reg:<5} in={in_rae_n:.4f} "
                  f"old_pred={old_str} new_pred={new_pred:.4f} "
                  f"band=[{band_lo:.4f}, {band_hi:.4f}] d={delta_use:+.4f}")
    except Exception as exc:  # pragma: no cover (best-effort)
        print(f"[warn] could not parse PRIMARY ladder: {exc}")

    summary = {
        "tag": TAG,
        "status": status,
        "global_delta_prior": GLOBAL_DELTA,
        "n_paired_total": n_pre + n_post,
        "n_pre": n_pre,
        "n_post": n_post,
        "shrinkage_tau2": tau2,
        "regime_estimates": {
            "PRE": {
                "n": n_pre,
                "mean_delta": round(m_pre, 6) if math.isfinite(m_pre) else None,
                "sd": round(s_pre, 6) if math.isfinite(s_pre) else None,
                "ci_95": [round(pre_ci[0], 6), round(pre_ci[1], 6)]
                          if math.isfinite(pre_ci[0]) else None,
                "shrunk_delta": round(pre_shrunk, 6),
                "weight_on_data": round(w_pre, 4),
            },
            "POST": {
                "n": n_post,
                "mean_delta": round(m_post, 6) if math.isfinite(m_post) else None,
                "sd": round(s_post, 6) if math.isfinite(s_post) else None,
                "ci_95": [round(post_ci[0], 6), round(post_ci[1], 6)]
                          if math.isfinite(post_ci[0]) else None,
                "shrunk_delta": round(post_shrunk, 6) if n_post > 0 else None,
                "weight_on_data": round(w_post, 4) if n_post > 0 else 0.0,
            },
        },
        "anchored_pairs": [
            {"label": lab, "in_RAE": round(ir, 4), "LB": round(lb, 4),
             "delta": round(lb - ir, 4), "regime": reg}
            for lab, ir, lb, reg in merged
        ],
        "primary_recalibrated": primary,
        "wall_sec": round(time.time() - t0, 2),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[save] {OUT_JSON}  ({summary['wall_sec']}s)")
    return summary


if __name__ == "__main__":
    main()

"""Scrape the OpenADMET PXR Gradio leaderboard and log our scores.

Usage:
    python scripts/log_lb_scores.py            # scrape both tracks and append a row each
    python scripts/log_lb_scores.py latest     # just print current LB best per track

Appends rows to data/processed/leaderboard_log.csv. New columns (nb1670):
    mae, rae, r2, spearman, kendall, submitted_utc_on_lb,
    lddt_pli, bisyrmsd, lddt_lp, coverage
Existing rows retain prior schema; missing cells become NaN.

The Gradio space typically only shows best-per-user; we still try the HF Spaces
queue API as a fallback to recover per-submission scores by ID.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "processed" / "leaderboard_log.csv"
SPACE_URL = "https://openadmet-pxr-challenge.hf.space"
USER_HANDLE = "xX-its-amit-Xx"
TRACKS = ("activity", "structure")
TRACK_API = {
    "activity": "/load_activity_leaderboard",
    "structure": "/load_structure_leaderboard",
}

# Column name mapping: Gradio header -> our snake_case log column.
# Headers vary slightly with unicode (rho/tau) so we normalise on a lowercase
# ASCII-folded prefix match. Activity columns first, then structure.
HEADER_TO_LOG = {
    "rank": "user_rank",
    "submitted": "submitted_utc_on_lb",
    "mae": "mae",
    "rae": "rae",
    "r2": "r2",
    "spearman": "spearman",
    "kendall": "kendall",
    # structure track
    "lddt-pli": "lddt_pli",
    "bisyrmsd": "bisyrmsd",
    "lddt-lp": "lddt_lp",
    "coverage": "coverage",
}

# Primary score for each track (used to populate legacy lb_score column).
TRACK_PRIMARY_HEADER = {
    "activity": "rae",
    "structure": "lddt-pli",
}

LEGACY_COLS = [
    "submitted_at", "model_stem", "local_oof_rae", "local_ratio",
    "leaderboard_rae", "leaderboard_rank", "note",
]
META_COLS = ["timestamp_utc", "track", "user_rank", "lb_score", "n_submissions_visible"]
NEW_COLS = [
    "mae", "rae", "r2", "spearman", "kendall", "submitted_utc_on_lb",
    "lddt_pli", "bisyrmsd", "lddt_lp", "coverage",
]
ALL_COLS = LEGACY_COLS + META_COLS + NEW_COLS


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _normalise_header(h: str) -> str:
    """Lowercase + strip unicode greek so 'Spearman ρ' -> 'spearman'."""
    s = str(h).strip().lower()
    # drop everything after the first space (handles 'spearman p', "kendall's t")
    s = s.split()[0] if s else s
    # strip trailing apostrophe-s
    if s.endswith("'s"):
        s = s[:-2]
    return s


def _rows_and_headers_from_payload(payload):
    """Return (headers, rows) where rows is list-of-lists.

    Handles the {'headers': [...], 'data': [...]} dict shape used by the
    OpenADMET space, plus a couple of older fallbacks.
    """
    if payload is None:
        return [], []
    if isinstance(payload, dict):
        headers = payload.get("headers") or []
        for key in ("data", "value", "rows", "records"):
            if key in payload:
                rows = payload[key]
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    headers = headers or list(rows[0].keys())
                    rows = [list(r.values()) for r in rows]
                return list(headers), list(rows or [])
        return list(headers), []
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            headers = list(payload[0].keys())
            return headers, [list(r.values()) for r in payload]
        return [], payload
    return [], []


def fetch_leaderboard(track: str):
    """Use gradio_client to hit the leaderboard endpoint for a track."""
    from gradio_client import Client

    client = Client(SPACE_URL, verbose=False)
    api = TRACK_API[track]
    try:
        result = client.predict(api_name=api)
    except Exception as e:
        print(f"[{track}] endpoint {api} failed: {e}")
        return [], []
    return _rows_and_headers_from_payload(result)


def _to_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def find_user_row(headers, rows, handle: str = USER_HANDLE):
    """Return dict of {log_col: value} pulled by header name. Rank falls back
    to row index + 1 if the header lookup fails."""
    if not rows:
        return None
    # Build normalised header -> column index
    norm_headers = [_normalise_header(h) for h in headers]
    col_idx = {nh: i for i, nh in enumerate(norm_headers)}

    for i, row in enumerate(rows):
        cells = [str(c) for c in row]
        if not any(handle.lower() in c.lower() for c in cells):
            continue

        out: dict = {}
        for norm_h, log_col in HEADER_TO_LOG.items():
            idx = col_idx.get(norm_h)
            if idx is None or idx >= len(row):
                continue
            val = row[idx]
            if log_col in ("user_rank",):
                try:
                    out[log_col] = int(val)
                except (TypeError, ValueError):
                    out[log_col] = None
            elif log_col == "submitted_utc_on_lb":
                out[log_col] = str(val) if val is not None else None
            else:
                out[log_col] = _to_float(val)
        if out.get("user_rank") is None:
            out["user_rank"] = i + 1
        return out
    return None


def try_queue_api(submission_id: str | None = None) -> dict | None:
    """Fallback: hit HF Spaces queue API for our own submission record."""
    if not submission_id:
        return None
    url = f"{SPACE_URL}/api/submissions/{submission_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def append_row(track: str, user_row: dict | None, n_visible: int) -> None:
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
    else:
        df = pd.DataFrame(columns=ALL_COLS)
    df = ensure_columns(df)

    new_row = {c: None for c in df.columns}
    new_row["timestamp_utc"] = utc_now()
    new_row["track"] = track
    new_row["n_submissions_visible"] = n_visible

    if user_row:
        for k, v in user_row.items():
            if k in df.columns:
                new_row[k] = v
        # populate legacy lb_score from the track's primary metric
        primary_log_col = HEADER_TO_LOG[TRACK_PRIMARY_HEADER[track]]
        new_row["lb_score"] = user_row.get(primary_log_col)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)


def scrape_and_log() -> dict:
    out = {}
    for track in TRACKS:
        headers, rows = fetch_leaderboard(track)
        user_row = find_user_row(headers, rows)
        append_row(track, user_row, len(rows))
        if user_row:
            primary = HEADER_TO_LOG[TRACK_PRIMARY_HEADER[track]]
            print(
                f"[{track:9s}] rank={user_row.get('user_rank')} "
                f"{primary}={user_row.get(primary)} "
                f"submitted={user_row.get('submitted_utc_on_lb')} "
                f"visible={len(rows)}"
            )
        else:
            print(f"[{track:9s}] user not found; visible={len(rows)}")
        out[track] = {"row": user_row, "n_visible": len(rows)}
    return out


def latest() -> None:
    """Print the most recent best per track from the log."""
    if not LOG_PATH.exists():
        print("No leaderboard_log.csv yet -- run without args first.")
        return
    df = pd.read_csv(LOG_PATH)
    df = ensure_columns(df)
    have = df.dropna(subset=["track", "lb_score"])
    if have.empty:
        print("Log has no LB scores yet."); return
    for track in TRACKS:
        sub = have[have["track"] == track].sort_values("timestamp_utc").tail(1)
        if sub.empty:
            print(f"[{track:9s}] no entries"); continue
        r = sub.iloc[0]
        primary = HEADER_TO_LOG[TRACK_PRIMARY_HEADER[track]]
        print(
            f"[{track:9s}] rank={r['user_rank']} {primary}={r.get(primary)} "
            f"as_of={r['timestamp_utc']} (visible={r['n_submissions_visible']})"
        )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scrape"
    if mode == "latest":
        latest()
    else:
        scrape_and_log()

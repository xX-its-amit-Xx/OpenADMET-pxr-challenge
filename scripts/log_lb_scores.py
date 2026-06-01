"""Scrape the OpenADMET PXR Gradio leaderboard and log our scores.

Usage:
    python scripts/log_lb_scores.py            # scrape both tracks and append a row each
    python scripts/log_lb_scores.py latest     # just print current LB best per track

Appends rows to data/processed/leaderboard_log.csv with columns:
    timestamp_utc, track, user_rank, lb_score, n_submissions_visible

The Gradio space typically only shows best-per-user; we still try the HF Spaces
queue API as a fallback to recover per-submission scores by ID.
"""
from __future__ import annotations

import json
import re
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
NEW_COLS = ["timestamp_utc", "track", "user_rank", "lb_score", "n_submissions_visible"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _rows_from_payload(payload) -> list[list]:
    """Pull a list-of-rows table out of whatever Gradio returns."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("data", "value", "rows", "records"):
            if key in payload:
                return _rows_from_payload(payload[key])
        return []
    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            return payload
        if payload and isinstance(payload[0], dict):
            return [list(r.values()) for r in payload]
    return []


def fetch_leaderboard(track: str) -> list[list]:
    """Use gradio_client to hit the leaderboard endpoint for a track."""
    from gradio_client import Client

    client = Client(SPACE_URL, verbose=False)
    api = TRACK_API[track]
    try:
        result = client.predict(api_name=api)
    except Exception as e:
        print(f"[{track}] endpoint {api} failed: {e}")
        return []
    return _rows_from_payload(result)


def find_user_row(rows: list[list], handle: str = USER_HANDLE):
    """Return (rank, score) or (None, None) — Gradio sometimes orders columns
    [rank, user, score] or [user, score] or [user, n_subs, best_score]."""
    for i, row in enumerate(rows):
        cells = [str(c) for c in row]
        if any(handle.lower() in c.lower() for c in cells):
            score = None
            rank = None
            for c in cells:
                try:
                    v = float(c)
                except (TypeError, ValueError):
                    continue
                if 0.0 < v < 5.0 and score is None:
                    score = v
                elif v.is_integer() and rank is None and 0 < v < 1000:
                    rank = int(v)
            if rank is None:
                rank = i + 1
            return rank, score
    return None, None


def try_queue_api(submission_id: str | None = None) -> dict | None:
    """Fallback: hit HF Spaces queue API for our own submission record.
    Returns parsed JSON or None.
    """
    if not submission_id:
        return None
    url = f"{SPACE_URL}/api/submissions/{submission_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def append_row(track: str, rank, score, n_visible: int) -> None:
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
    else:
        df = pd.DataFrame(columns=NEW_COLS)
    df = ensure_columns(df)
    new_row = {c: None for c in df.columns}
    new_row.update({
        "timestamp_utc": utc_now(),
        "track": track,
        "user_rank": rank,
        "lb_score": score,
        "n_submissions_visible": n_visible,
    })
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)


def scrape_and_log() -> dict:
    out = {}
    for track in TRACKS:
        rows = fetch_leaderboard(track)
        rank, score = find_user_row(rows)
        append_row(track, rank, score, len(rows))
        out[track] = {"rank": rank, "score": score, "n_visible": len(rows)}
        print(f"[{track:9s}] rank={rank} score={score} visible={len(rows)}")
    return out


def latest() -> None:
    """Print the most recent best per track from the log."""
    if not LOG_PATH.exists():
        print("No leaderboard_log.csv yet — run without args first.")
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
        print(f"[{track:9s}] rank={r['user_rank']} score={r['lb_score']} "
              f"as_of={r['timestamp_utc']} (visible={r['n_submissions_visible']})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scrape"
    if mode == "latest":
        latest()
    else:
        scrape_and_log()

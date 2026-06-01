"""Scrape full activity + structure leaderboards, filter to our handle,
cross-reference with submission_log.csv, save to lb_history.csv."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "lb_history.csv"
SPACE_URL = "https://openadmet-pxr-challenge.hf.space"
HANDLE = "xX-its-amit-Xx"

TRACK_API = {
    "activity": "/load_activity_leaderboard",
    "structure": "/load_structure_leaderboard",
}


def fetch_lb(track: str):
    from gradio_client import Client
    client = Client(SPACE_URL, verbose=False)
    res = client.predict(api_name=TRACK_API[track])
    return res


def normalize_rows(payload):
    """Pull a list of dicts / list of lists out of whatever Gradio returns."""
    if payload is None:
        return [], []
    # dict with 'headers' and 'data' is typical for gr.Dataframe
    if isinstance(payload, dict):
        headers = payload.get("headers") or payload.get("columns")
        data = payload.get("data") or payload.get("value") or payload.get("rows")
        if data is None:
            for k in payload:
                if isinstance(payload[k], list):
                    data = payload[k]; break
        if isinstance(data, list) and data and isinstance(data[0], list):
            return headers or [], data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            hdr = list(data[0].keys())
            return hdr, [[r.get(h) for h in hdr] for r in data]
        return headers or [], data or []
    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            return [], payload
        if payload and isinstance(payload[0], dict):
            hdr = list(payload[0].keys())
            return hdr, [[r.get(h) for h in hdr] for r in payload]
    return [], []


def main():
    rows_all = []
    for track in ("activity", "structure"):
        try:
            payload = fetch_lb(track)
        except Exception as e:
            print(f"[{track}] fetch failed: {e}")
            continue
        headers, data = normalize_rows(payload)
        print(f"[{track}] headers={headers}")
        print(f"[{track}] {len(data)} rows")
        if data:
            print(f"[{track}] sample row: {data[0]}")
        for i, row in enumerate(data):
            rec = {"track": track, "rank_overall": i + 1}
            if headers and len(headers) == len(row):
                for h, v in zip(headers, row):
                    rec[str(h)] = v
            else:
                for j, v in enumerate(row):
                    rec[f"col{j}"] = v
            rows_all.append(rec)
    df = pd.DataFrame(rows_all)
    df["scraped_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # filter to our handle
    text_cols = [c for c in df.columns if df[c].dtype == object]
    mask = pd.Series(False, index=df.index)
    for c in text_cols:
        mask = mask | df[c].astype(str).str.contains(HANDLE, case=False, na=False)
    ours = df[mask].copy()
    print("\n=== OUR ROWS ===")
    print(ours.to_string())

    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT}")
    ours_path = PROCESSED / "lb_history_ours.csv"
    ours.to_csv(ours_path, index=False)
    print(f"Saved {len(ours)} of our rows -> {ours_path}")


if __name__ == "__main__":
    main()

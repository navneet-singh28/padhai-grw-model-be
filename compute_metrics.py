#!/usr/bin/env python
"""
compute_metrics.py
==================

Examples
--------
1) Analyse local parquet:
   python compute_metrics.py opens.parquet \
          --start 2025-03-01 --end 2025-04-20

2) Pull data straight from Mixpanel (no local files):
   python compute_metrics.py \
          --fetch --from-date 2024-01-01 --to-date 2025-04-20 \
          --event app_opened --start 2025-03-01 --end 2025-04-20

Environment
-----------
• set MIXPANEL_API_SECRET  (or pass --api-secret)  for Export API access
• dependencies: pandas, numpy, tqdm, python-dotenv, requests
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:  # fall back to noop if tqdm not installed
    def tqdm(x, **kwargs):  # type: ignore
        return x


# ──────────────────────────────────────────────────────────────────────────────
# CLI                                                                       
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("offline_all_metrics.py")

    # either local files … -----------------------------------------------------
    p.add_argument("events", nargs="*", type=Path,
                   help="One or more local event files (csv / parquet / …).")
    # … or remote fetch. -------------------------------------------------------
    p.add_argument("--fetch", action="store_true",
                   help="Pull events via Mixpanel Export API instead of "
                        "reading local files.")
    p.add_argument("--from-date", default=None,
                   help="Export-API: earliest date (YYYY-MM-DD).")
    p.add_argument("--to-date", default=None,
                   help="Export-API: latest date (YYYY-MM-DD).")
    p.add_argument("--event", default="app_opened",
                   help="Export-API: event name to download.")
    p.add_argument("--api-secret", default=None,
                   help="Mixpanel API secret. If omitted we read "
                        "MIXPANEL_API_SECRET from env/.env.")

    # analysis window ----------------------------------------------------------
    p.add_argument("--pre-window-start", default="2020-01-01",
                   help="Earliest date to keep after loading (YYYY-MM-DD).")
    p.add_argument("--start", required=True,
                   help="Analysis window start date (YYYY-MM-DD).")
    p.add_argument("--end", required=True,
                   help="Analysis window end   date (YYYY-MM-DD).")

    # column overrides ---------------------------------------------------------
    p.add_argument("--user-col", default=None,
                   help="Override user id column name (default "
                        "auto-detect user_id/distinct_id).")
    p.add_argument("--time-col", default=None,
                   help="Override timestamp column (default "
                        "auto-detect time/timestamp).")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Export-API download                                                       
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_mixpanel_events(api_secret: str,
                           from_date: str,
                           to_date: str,
                           event: str) -> pd.DataFrame:
    """
    Stream `app_opened` (or any) events from Mixpanel Export-API,
    return DataFrame[user_id, date].
    """
    url = (
        "https://data.mixpanel.com/api/2.0/export/"
        f"?from_date={from_date}"
        f"&to_date={to_date}"
        f"&event=[\"{event}\"]"
        "&format=json"
    )

    print(f"[INFO] Fetching events via Export API: {event} {from_date} → {to_date}")
    resp = requests.get(url, auth=(api_secret, ""), stream=True)
    resp.raise_for_status()

    rows = []
    for line in tqdm(resp.iter_lines(decode_unicode=True), desc="download"):
        if not line:
            continue
        obj = json.loads(line)
        props = obj.get("properties", {})
        rows.append({
            "user_id": str(props.get("distinct_id")),
            "time": props.get("time"),           # ms or s epoch
        })

    if not rows:
        sys.exit("[ERROR] Export API returned 0 rows.")

    df = pd.DataFrame(rows)
    return _clean_and_bucket_dates(df, "user_id", "time")


# ──────────────────────────────────────────────────────────────────────────────
# Local-file loader                                                         
# ──────────────────────────────────────────────────────────────────────────────
_READERS = {
    ".csv":     pd.read_csv,
    ".tsv":     lambda p: pd.read_csv(p, sep="\t"),
    ".parquet": pd.read_parquet,
    ".pq":      pd.read_parquet,
    ".feather": pd.read_feather,
    ".json":    pd.read_json,
}


def _load_one(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")
    ext = path.suffix.lower()
    if ext not in _READERS:
        sys.exit(f"[ERROR] Unsupported extension {ext}")
    return _READERS[ext](path)


def _load_local_events(files: Iterable[Path],
                       user_col: str | None,
                       time_col: str | None) -> pd.DataFrame:
    dfs = [_load_one(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # resolve column names -----------------------------------------------------
    user_col = user_col or (
        "user_id" if "user_id" in df.columns else
        "distinct_id" if "distinct_id" in df.columns else None
    )
    if user_col is None:
        sys.exit("[ERROR] Could not find user column; use --user-col")

    time_col = time_col or (
        "time" if "time" in df.columns else
        "timestamp" if "timestamp" in df.columns else None
    )
    if time_col is None:
        sys.exit("[ERROR] Could not find timestamp column; use --time-col")

    return _clean_and_bucket_dates(df, user_col, time_col)


# ──────────────────────────────────────────────────────────────────────────────
# Shared cleaning util                                                     
# ──────────────────────────────────────────────────────────────────────────────
def _clean_and_bucket_dates(df: pd.DataFrame,
                            uid_col: str,
                            ts_col: str) -> pd.DataFrame:
    """
    • timestamps → UTC calendar-day
    • drop duplicates  (user × day)
    """
    ts = df[ts_col]
    if np.issubdtype(ts.dtype, np.number):
        # Heuristic ms vs s
        unit = "ms" if ts.iloc[0] > 1e11 else "s"
        dates = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        dates = pd.to_datetime(ts, utc=True, errors="coerce")

    dates = dates.dt.floor("D").dt.tz_localize(None)

    out = (
        pd.DataFrame({
            "user_id": df[uid_col].astype(str),
            "date": dates
        })
        .dropna()
        .drop_duplicates()
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# classifyState()  (literal port from JS)                                   
# ──────────────────────────────────────────────────────────────────────────────
DAY = timedelta(days=1)


def classify_state(day: datetime,
                   first_day: datetime,
                   active: set[datetime]) -> str:
    ds = (day - first_day).days
    if ds <= 3:
        return "New User"

    # trailing-window counts (excluding today)
    c3 = c7 = 0
    for i in range(1, 8):
        past = day - i * DAY
        if past in active:
            c7 += 1
            if i <= 3:
                c3 += 1

    # Resurrected?
    dormant_before = all(
        (day - i * DAY) not in active
        for i in range(3, 10)
    )
    recent_active = ((day - DAY) in active) or ((day - 2 * DAY) in active)
    if recent_active and dormant_before:
        return "Resurrected"

    if c3 >= 2:
        return "Current"
    if c3 == 1 and c7 >= 2:
        return "At-risk DAU"
    if c7 == 1:
        return "At-risk WAU"
    return "Dormant"


# ──────────────────────────────────────────────────────────────────────────────
# Metric engine                                                            
# ──────────────────────────────────────────────────────────────────────────────
def _safe(num: int, den: int) -> float:
    return num / den if den else 0.0


def compute_metrics(df: pd.DataFrame,
                    start: datetime,
                    end: datetime) -> dict[str, float]:
    users: dict[str, set[datetime]] = defaultdict(set)
    for u, d in zip(df.user_id, df.date):
        users[u].add(d)

    cnt = Counter(
        nurr_numerator=0, nurr_denominator=0,
        curr_numerator=0, curr_denominator=0,
        dau_loss_numerator=0,
        wau_loss_numerator=0, wau_loss_denominator=0,
        dormancy_numerator=0, dormancy_denominator=0,
        resurrection_numerator=0, resurrection_denominator=0,
        surr_numerator=0, surr_denominator=0,
    )

    day_count = int((end - start).days) + 1
    for active in tqdm(users.values(), desc="analyse"):
        first = min(active)
        state_prev = None
        for o in range(day_count):
            day = start + o * DAY
            if day < first:
                continue

            state = classify_state(day, first, active)

            # ------------------ metrics --------------------------------------
            if state == "New User" and day == first:
                cnt["nurr_denominator"] += 1
                day4 = first + 4 * DAY
                if day4 <= end and classify_state(day4, first, active) == "Current":
                    cnt["nurr_numerator"] += 1

            if state_prev is not None:
                if state_prev == "Current":
                    cnt["curr_denominator"] += 1
                    if state == "Current":
                        cnt["curr_numerator"] += 1
                    if state == "At-risk DAU":
                        cnt["dau_loss_numerator"] += 1

                if state_prev == "At-risk DAU":
                    cnt["wau_loss_denominator"] += 1
                    if state == "At-risk WAU":
                        cnt["wau_loss_numerator"] += 1

                if state_prev == "At-risk WAU":
                    cnt["dormancy_denominator"] += 1
                    if state == "Dormant":
                        cnt["dormancy_numerator"] += 1

                if state_prev == "Dormant":
                    cnt["resurrection_denominator"] += 1
                    if state == "Resurrected":
                        cnt["resurrection_numerator"] += 1

                if state_prev == "Resurrected":
                    cnt["surr_denominator"] += 1
                    if state == "Current":
                        cnt["surr_numerator"] += 1
            # -----------------------------------------------------------------
            state_prev = state

    return {
        "NURR":              _safe(cnt["nurr_numerator"], cnt["nurr_denominator"]),
        "CURR":              _safe(cnt["curr_numerator"], cnt["curr_denominator"]),
        "DAU_loss":          _safe(cnt["dau_loss_numerator"], cnt["curr_denominator"]),
        "WAU_loss":          _safe(cnt["wau_loss_numerator"], cnt["wau_loss_denominator"]),
        "Dormancy_rate":     _safe(cnt["dormancy_numerator"], cnt["dormancy_denominator"]),
        "Resurrection_rate": _safe(cnt["resurrection_numerator"], cnt["resurrection_denominator"]),
        "SURR":              _safe(cnt["surr_numerator"], cnt["surr_denominator"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main                                                                     
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    # dates --------------------------------------------------------------------
    pre_window = pd.Timestamp(args.pre_window_start)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        sys.exit("[ERROR] --end must be >= --start")

    # load or fetch events -----------------------------------------------------
    if args.fetch:
        load_dotenv()
        secret = args.api_secret or os.getenv("MIXPANEL_API_SECRET")
        if not secret:
            sys.exit("[ERROR] Need --api-secret or env MIXPANEL_API_SECRET for --fetch")
        if not args.from_date or not args.to_date:
            sys.exit("[ERROR] --from-date and --to-date required with --fetch")

        events = _fetch_mixpanel_events(secret, args.from_date,
                                        args.to_date, args.event)
    else:
        if not args.events:
            sys.exit("[ERROR] Provide event files OR use --fetch")
        events = _load_local_events(args.events, args.user_col, args.time_col)

    # clip to analysis horizon + look-back window
    events = events.query("date >= @pre_window & date <= @end")

    metrics = compute_metrics(events, start, end)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

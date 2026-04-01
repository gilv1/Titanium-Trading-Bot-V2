"""Quick diagnostic for futures session regime quality.

Focuses on the NY regular session and quantifies:
- Opening impulse (first 15 minutes)
- 60-minute follow-through
- Intraday range and close direction
- Whipsaw score (opening move vs one-hour move disagreement)

Usage:
    python analysis/futures_regime_review.py --csv data/historical/MES.csv --month 2026-03
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd


@dataclass
class DayRegime:
    date: str
    open_price: float
    close_price: float
    day_change: float
    day_range: float
    first15_change: float
    first15_range: float
    first60_change: float
    whipsaw: bool


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df["time"] = df["time"].dt.tz_convert("America/New_York")
    return df.sort_values("time")


def _regular_session(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime("09:30").time()
    end = pd.to_datetime("16:00").time()
    return df[(df["time"].dt.time >= start) & (df["time"].dt.time <= end)].copy()


def _build_rows(df: pd.DataFrame) -> list[DayRegime]:
    rows: list[DayRegime] = []
    for d, g in df.groupby(df["time"].dt.date):
        g = g.sort_values("time")
        if len(g) < 61:
            continue

        open_price = float(g.iloc[0]["open"])
        close_price = float(g.iloc[-1]["close"])

        first15 = g[g["time"].dt.time <= pd.to_datetime("09:45").time()]
        first60 = g[g["time"].dt.time <= pd.to_datetime("10:30").time()]

        first15_change = float(first15.iloc[-1]["close"] - open_price)
        first60_change = float(first60.iloc[-1]["close"] - open_price)

        whipsaw = (first15_change == 0 and first60_change != 0) or (
            first15_change * first60_change < 0
        )

        rows.append(
            DayRegime(
                date=str(d),
                open_price=open_price,
                close_price=close_price,
                day_change=float(close_price - open_price),
                day_range=float(g["high"].max() - g["low"].min()),
                first15_change=first15_change,
                first15_range=float(first15["high"].max() - first15["low"].min()),
                first60_change=first60_change,
                whipsaw=whipsaw,
            )
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Review intraday regime quality for futures data.")
    p.add_argument("--csv", default="data/historical/MES.csv", help="Path to OHLCV CSV.")
    p.add_argument("--month", default="2026-03", help="Month filter in YYYY-MM format.")
    args = p.parse_args()

    df = _load(args.csv)
    month_mask = df["time"].dt.strftime("%Y-%m") == args.month
    df = _regular_session(df[month_mask])

    rows = _build_rows(df)
    if not rows:
        print(f"No data available for month {args.month}")
        return

    out = pd.DataFrame([r.__dict__ for r in rows]).sort_values("date")
    numeric_cols = [
        "day_change",
        "day_range",
        "first15_change",
        "first15_range",
        "first60_change",
    ]
    out[numeric_cols] = out[numeric_cols].round(2)

    print("=== Daily Regime Summary ===")
    print(out[["date", *numeric_cols, "whipsaw"]].to_string(index=False))

    whipsaw_ratio = out["whipsaw"].mean() * 100.0
    print("\n=== Aggregate ===")
    print(f"Days analyzed: {len(out)}")
    print(f"Whipsaw days: {int(out['whipsaw'].sum())}/{len(out)} ({whipsaw_ratio:.1f}%)")
    print(out[numeric_cols].mean().round(2).to_string())

    recommendation = (
        "Delay entries to >= 09:45 ET on whipsaw-prone days."
        if whipsaw_ratio >= 40
        else "Opening entries are relatively stable; 09:30 participation is acceptable."
    )
    print(f"\nRecommendation: {recommendation}")


if __name__ == "__main__":
    main()

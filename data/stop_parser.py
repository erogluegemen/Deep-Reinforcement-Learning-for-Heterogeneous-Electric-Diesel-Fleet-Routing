"""
Stop parser: loads the DHL PuD stop export and returns a clean DataFrame.

PUD Fac → internal depot mapping:
  GTW → IGA  (Sabiha Gökçen area routes start IS*)
  EAT → SAW  (Istanbul Airport area routes start EA*)
  CET → CET  (Central Istanbul routes start CE*)

Only rows with Act Ckpt Code in {OK, PU} are kept as valid completed stops.
Rows missing lat/lgtd are dropped.
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP_FILE = os.path.join(ROOT, "data", "raw", "Copy of PuD_STOP_DetailExport_cleaned.xlsm")

FAC_TO_DEPOT = {"GTW": "IGA", "EAT": "SAW", "CET": "CET"}

VALID_CODES = {"OK", "PU"}


def _parse_time_hhmm(val):
    """Convert 'HH:MM' string to minutes from midnight. Returns None if invalid."""
    if pd.isna(val) or str(val).strip() == "":
        return None
    s = str(val).strip()
    try:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def load_stop_data():
    """
    Load and clean the RoadMap sheet.

    Returns a DataFrame with columns:
      date, route, depot, lat, lng, demand_kg,
      open_min, close_min, act_time_min, prod_code, act_code
    """
    raw = pd.read_excel(STOP_FILE, sheet_name="RoadMap", engine="openpyxl")

    # Drop rows with missing coordinates
    df = raw.dropna(subset=["lat", "lgtd"]).copy()

    # Keep only valid completed stops
    df = df[df["Act Ckpt Code"].isin(VALID_CODES)].copy()

    # Parse date
    df["date"] = pd.to_datetime(df["Act Dt"].astype(str), format="%Y%m%d")

    # Parse actual service time (HH:MM → minutes from midnight)
    df["act_time_min"] = df["Act Tm"].apply(_parse_time_hhmm)

    # Parse time windows
    df["open_min"] = df["Open"].apply(
        lambda v: 0 if pd.isna(v) or str(v).strip() == "" else _parse_time_hhmm(v) or 0
    )
    df["close_min"] = df["Closed"].apply(
        lambda v: 1439 if pd.isna(v) or str(v).strip() in ("", "23:59")
        else _parse_time_hhmm(v) or 1439
    )

    # Map depot
    df["depot"] = df["PUD Fac"].map(FAC_TO_DEPOT)
    df = df.dropna(subset=["depot"]).copy()

    # Use Weight column; fill missing with 0.5 kg (very light parcel assumption)
    df["demand_kg"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.5)

    out = df.rename(columns={
        "PUD Rte": "route",
        "lat": "lat",
        "lgtd": "lng",
        "Prod Code": "prod_code",
        "Act Ckpt Code": "act_code",
    })[[
        "date", "route", "depot", "lat", "lng",
        "demand_kg", "open_min", "close_min", "act_time_min",
        "prod_code", "act_code",
    ]]

    return out.reset_index(drop=True)


if __name__ == "__main__":
    df = load_stop_data()
    print(f"Cleaned stops: {len(df)}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Depot distribution:\n{df['depot'].value_counts()}")
    print(f"Null act_time_min: {df['act_time_min'].isna().sum()}")
    print(f"Time window sample:\n{df[['open_min','close_min']].describe()}")
    print(f"\nSample rows:")
    print(df.head(3).to_string())

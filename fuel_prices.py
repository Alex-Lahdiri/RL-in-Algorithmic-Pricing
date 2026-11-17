"""Collect UK pump prices from the retailers' published price feeds.

The large UK fuel retailers each publish a JSON file of their current
station-level prices. This script pulls every feed, keeps the Greater London
stations and saves one timestamped snapshot. Run on a timer, the snapshots
build a panel of prices per station.

Usage
    python fuel_prices.py           one snapshot
    python fuel_prices.py --poll    one snapshot every six hours until interrupted
    python fuel_prices.py --status  what has been collected so far
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Imperial Business School FYC academic project)"}

RETAILER_URLS = {
    "Applegreen": "https://applegreenstores.com/fuel-prices/data.json",
    "Ascona":     "https://fuelprices.asconagroup.co.uk/newfuel.json",
    "Asda":       "https://storelocator.asda.com/fuel_prices_data.json",
    "BP":         "https://www.bp.com/en_gb/united-kingdom/home/fuelprices/fuel_prices_data.json",
    "Esso":       "https://fuelprices.esso.co.uk/latestdata.json",
    "JET":        "https://jetlocal.co.uk/fuel_prices_data.json",
    "Karan":      "https://api2.krlmedia.com/integration/live_price/krl",
    "Morrisons":  "https://www.morrisons.com/fuel-prices/fuel.json",
    "Moto":       "https://moto-way.com/fuel-price/fuel_prices.json",
    "MFG":        "https://fuel.motorfuelgroup.com/fuel_prices_data.json",
    "Rontec":     "https://www.rontec-servicestations.co.uk/fuel-prices/data/fuel_prices_data.json",
    "Sainsburys": "https://api.sainsburys.co.uk/v1/exports/latest/fuel_prices_data.json",
    "SGN":        "https://www.sgnretail.uk/files/data/SGN_daily_fuel_prices.json",
    "Tesco":      "https://www.tesco.com/fuel_prices/fuel_prices_data.json",
}

LONDON_PREFIXES = {"E", "EC", "N", "NW", "SE", "SW", "W", "WC"}

SNAPSHOT_DIR = Path("fuel_snapshots")
INTERVAL_HOURS = 6
MAX_FEED_AGE_DAYS = 7


def fetch_retailer(brand, url, timeout=30):
    """Fetch one retailer's JSON. Returns list of station rows."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json() if "json" in r.headers.get("content-type", "") else json.loads(r.text)
    except Exception as e:
        print(f"  [FAIL] {brand}: {type(e).__name__}: {e}")
        return []

    rows = []
    for s in data.get("stations", []):
        loc = s.get("location") or {}
        prices = s.get("prices") or {}
        try:
            lat = float(loc.get("latitude", "nan"))
            lon = float(loc.get("longitude", "nan"))
        except (TypeError, ValueError):
            lat, lon = np.nan, np.nan
        rows.append({
            "retailer": brand,
            "site_id": str(s.get("site_id", "")),
            "brand": s.get("brand", brand),
            "address": s.get("address", ""),
            "postcode": str(s.get("postcode", "")).upper().strip(),
            "lat": lat, "lon": lon,
            "E5":  prices.get("E5"),
            "E10": prices.get("E10"),
            "B7":  prices.get("B7"),
            "SDV": prices.get("SDV"),
            "source_last_updated": data.get("last_updated", ""),
        })
    print(f"  [OK]   {brand}: {len(rows)} stations")
    return rows


def pull_snapshot():
    print(f"Pulling at {datetime.now().isoformat(timespec='seconds')}")
    rows = []
    for brand, url in RETAILER_URLS.items():
        rows.extend(fetch_retailer(brand, url))
    df = pd.DataFrame(rows)
    df["pulled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df


def filter_region(df, prefixes=LONDON_PREFIXES):
    pc = df["postcode"].fillna("")
    outward = pc.str.extract(r"^([A-Z]+)")[0]
    return df[outward.isin(prefixes)].copy()


def save_snapshot(df, dirpath=SNAPSHOT_DIR):
    dirpath.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = dirpath / f"snapshot_{ts}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return path


def load_panel(dirpath=SNAPSHOT_DIR, drop_stale=True):
    """All snapshots as one table, sorted by station and time.

    drop_stale removes rows whose retailer feed was more than a week old when
    it was pulled. A retailer can leave an old file in place, and a price that
    never changes looks exactly like a station that has stopped competing.
    """
    paths = sorted(dirpath.glob("snapshot_*.csv"))
    if not paths:
        raise SystemExit(f"No snapshots in {dirpath}/. Run: python fuel_prices.py")
    panel = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    panel["pulled_at"] = pd.to_datetime(panel["pulled_at"], utc=True)
    if drop_stale:
        panel["source_last_updated"] = pd.to_datetime(
            panel["source_last_updated"], dayfirst=True, errors="coerce", utc=True)
        fresh = (panel["pulled_at"] - panel["source_last_updated"]).dt.days <= MAX_FEED_AGE_DAYS
        panel = panel[fresh].copy()
    return panel.sort_values(["site_id", "pulled_at"]).reset_index(drop=True)


def status():
    paths = sorted(SNAPSHOT_DIR.glob("snapshot_*.csv"))
    print(f"Snapshots collected: {len(paths)}")
    if not paths:
        return
    print(f"First: {paths[0].name}")
    print(f"Last:  {paths[-1].name}")
    panel = load_panel(drop_stale=False)
    print(f"Panel: {len(panel)} rows, "
          f"{panel['site_id'].nunique()} stations, "
          f"{panel['pulled_at'].nunique()} timestamps")


def collect_once():
    full = pull_snapshot()
    if "retailer" not in full:
        print("No retailer feed could be read; nothing saved.")
        return None
    print(f"\nTotal: {len(full)} stations across {full['retailer'].nunique()} retailers")
    local = filter_region(full)
    print(f"Region: {len(local)} stations")
    return save_snapshot(local)


def poll():
    print(f"Polling every {INTERVAL_HOURS}h. Interrupt to stop.")
    while True:
        try:
            collect_once()
        except KeyboardInterrupt:
            print("Stopped."); break
        except Exception as e:
            print(f"  pull failed, will retry next cycle: {e}")
        print(f"  sleeping {INTERVAL_HOURS}h\n")
        try:
            time.sleep(INTERVAL_HOURS * 3600)
        except KeyboardInterrupt:
            print("Stopped."); break


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--poll" in sys.argv:
        poll()
    else:
        collect_once()

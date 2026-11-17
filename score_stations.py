"""Score real London diesel prices with the classifiers trained on simulations.

Needs the snapshots from fuel_prices.py and the two models from classifier.py.

    stations   each station's diesel (B7) price series, scored by the
               single-series classifier
    pairs      every two stations within 2 km of each other, scored by the
               pair classifier

A high score means a price series resembles the simulated collusive runs. It
is not evidence of collusion: two stations owned by the same retailer, or two
stations passing through the same wholesale cost change, produce the same
pattern.

Usage
    python score_stations.py
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from classifier import extract_features, extract_pair_features
from fuel_prices import load_panel

warnings.filterwarnings("ignore", category=RuntimeWarning)

FIG_DIR = Path("figures")
FUEL = "B7"                      # diesel
MIN_OBS = 5
MAX_DIST_KM = 2.0
HIGH_SCORE_THRESHOLD = 0.5
LON_RANGE = (-0.55, 0.35)        # Greater London bounding box
LAT_RANGE = (51.25, 51.75)


def load_model(path):
    if not Path(path).exists():
        raise SystemExit(f"{path} not found. Run: python classifier.py")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(h))


def station_series(panel):
    """Per-station diesel price series, and where each station is."""
    series, meta = {}, {}
    for site, group in panel.groupby("site_id"):
        g = group.sort_values("pulled_at")
        prices = g[FUEL].dropna().values
        if len(prices) >= MIN_OBS:
            series[site] = prices
            meta[site] = {
                "retailer": g["retailer"].iloc[0],
                "postcode": g["postcode"].iloc[0],
                "lat":      g["lat"].iloc[0],
                "lon":      g["lon"].iloc[0],
            }
    return series, meta


def score_stations(series, meta, clf, cols):
    rows = []
    for site, prices in series.items():
        feats = extract_features(prices)
        if feats is None:
            continue
        feats.update({"site_id": site, "n_obs": len(prices), **meta[site]})
        rows.append(feats)
    df = pd.DataFrame(rows)
    df["collusion_score"] = clf.predict_proba(df[cols].values)[:, 1]
    return df


def score_pairs(series, meta, clf, cols):
    sites = list(series.keys())
    rows = []
    for i, site_a in enumerate(sites):
        ma = meta[site_a]
        for site_b in sites[i+1:]:
            mb = meta[site_b]
            d = haversine_km(ma["lat"], ma["lon"], mb["lat"], mb["lon"])
            if d > MAX_DIST_KM:
                continue
            a, b = series[site_a], series[site_b]
            n = min(len(a), len(b))
            feats = extract_pair_features(a[-n:], b[-n:])
            if feats is None:
                continue
            feats.update({
                "site_a": site_a, "site_b": site_b,
                "retailer_a": ma["retailer"], "retailer_b": mb["retailer"],
                "lat_a": ma["lat"], "lon_a": ma["lon"],
                "lat_b": mb["lat"], "lon_b": mb["lon"],
                "distance_km": d, "n_obs": n,
            })
            rows.append(feats)
    df = pd.DataFrame(rows)
    if len(df):
        df["collusion_score"] = clf.predict_proba(df[cols].values)[:, 1]
    return df


def plot_histogram(scores, xlabel, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=20, edgecolor="black", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, name)


def plot_map(meta, pairs_df, name="fig3_london_map.png"):
    stations_df = pd.DataFrame.from_dict(meta, orient="index").reset_index()
    stations_df.columns = ["site_id", "retailer", "postcode", "lat", "lon"]
    stations_df = stations_df[
        stations_df["lat"].between(*LAT_RANGE) &
        stations_df["lon"].between(*LON_RANGE)
    ].copy()
    print(f"Valid London stations: {len(stations_df)}")

    valid = (
        pairs_df["lat_a"].between(*LAT_RANGE) &
        pairs_df["lon_a"].between(*LON_RANGE) &
        pairs_df["lat_b"].between(*LAT_RANGE) &
        pairs_df["lon_b"].between(*LON_RANGE)
    )
    pdf = pairs_df[valid].copy()
    print(f"Valid pairs: {len(pdf)}")

    high_pairs = pdf[pdf["collusion_score"] > HIGH_SCORE_THRESHOLD]
    low_pairs  = pdf[pdf["collusion_score"] <= HIGH_SCORE_THRESHOLD]

    fig, ax = plt.subplots(figsize=(11, 9))

    for _, r in low_pairs.iterrows():
        ax.plot([r["lon_a"], r["lon_b"]], [r["lat_a"], r["lat_b"]],
                color="lightgray", lw=0.5, alpha=0.6, zorder=2)

    for _, r in high_pairs.iterrows():
        ax.plot([r["lon_a"], r["lon_b"]], [r["lat_a"], r["lat_b"]],
                color="crimson", lw=2.5, alpha=0.9, zorder=3)

    ax.scatter(stations_df["lon"], stations_df["lat"],
               s=45, c="black", alpha=0.85, zorder=4,
               edgecolor="white", linewidth=0.6)

    ax.set_xlim(*LON_RANGE)
    ax.set_ylim(*LAT_RANGE)

    try:                                   # street map underneath, if available
        import contextily as ctx
        ctx.add_basemap(ax, crs="EPSG:4326",
                        source=ctx.providers.CartoDB.Positron, zoom=11)
    except Exception as e:
        print(f"  no basemap ({type(e).__name__}); drawing stations only")
        ax.set_aspect(1 / np.cos(np.radians(51.5)))   # London-correct aspect ratio

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Greater London petrol stations\n"
        f"{len(high_pairs)} of {len(pdf)} station pairs within {MAX_DIST_KM:g}km "
        f"score > {HIGH_SCORE_THRESHOLD}"
    )

    legend = [
        Line2D([0], [0], color="crimson", lw=2.5,
               label=f"Pair score > {HIGH_SCORE_THRESHOLD}"),
        Line2D([0], [0], color="lightgray", lw=0.7,
               label="Pair score below threshold"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=8, label="Petrol station"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    _save(fig, name)


def _save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {FIG_DIR / name}")
    plt.close(fig)


def main():
    clf, cols = load_model("trained_classifier.pkl")
    clf_pair, pair_cols = load_model("trained_classifier_pair.pkl")

    panel = load_panel(drop_stale=True)
    print(f"After staleness filter: {len(panel)} rows, "
          f"{panel['site_id'].nunique()} stations")

    series, meta = station_series(panel)
    print(f"Stations with >={MIN_OBS} obs: {len(series)}")
    if not series:
        raise SystemExit("Not enough observations per station yet. Collect more snapshots.")

    real_df = score_stations(series, meta, clf, cols)
    print(f"\nScored {len(real_df)} stations")
    print(real_df["collusion_score"].describe())
    plot_histogram(real_df["collusion_score"],
                   "Collusion score (0 = competitive, 1 = collusive)", "Number of stations",
                   f"Classifier scores across {len(real_df)} Greater London stations",
                   "station_scores.png")

    pairs_df = score_pairs(series, meta, clf_pair, pair_cols)
    print(f"\nPairs within {MAX_DIST_KM}km: {len(pairs_df)}")
    if not len(pairs_df):
        return
    print("\nScore distribution:")
    print(pairs_df["collusion_score"].describe())
    plot_histogram(pairs_df["collusion_score"], "Pair collusion score", "Number of pairs",
                   f"Scores across {len(pairs_df)} station pairs within {MAX_DIST_KM}km",
                   "pair_scores.png")

    print("\nTop 10 highest-scoring pairs:")
    print(pairs_df.sort_values("collusion_score", ascending=False)
          .head(10)[["retailer_a", "retailer_b", "distance_km", "n_obs", "collusion_score"]])

    pairs_df.to_csv("pair_scores.csv", index=False)
    print("Wrote pair_scores.csv")
    plot_map(meta, pairs_df)


if __name__ == "__main__":
    main()

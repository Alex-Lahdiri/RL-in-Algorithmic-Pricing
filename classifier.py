"""Train classifiers that separate collusive from competitive price paths.

Input: the labelled trajectories written by `python simulator.py trajectories`.
A run is labelled collusive when its markup index (delta) is above 0.5.

Two gradient-boosting models are trained, each on seven scale-invariant
features so that the level of prices never matters, only their shape:

    single-series   one firm's price path
    pair            how two firms' price paths move together

Validation is grouped. Every run contributes two single-series rows, one per
firm, and the 48 runs come from only six parameter settings. A plain shuffled
split therefore tests the model on rows whose partner firm, or whose sibling
runs from the same setting, it has already trained on. The script reports the
out-of-fold ROC AUC three ways so the size of that effect is visible:

    rows shuffled                 the split first used; leaks
    whole runs held out           both firms of a run stay on one side
    whole settings held out       every run from one (alpha, beta) cell stays on one side

Usage
    python classifier.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    cross_val_predict, cross_val_score,
    LeaveOneGroupOut, StratifiedKFold, StratifiedGroupKFold,
)
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

TRAJ_DIR = Path("trajectories_perturbed")
FIG_DIR = Path("figures")

FEATURE_COLS = [
    "norm_level", "coef_variation", "mean_abs_change",
    "std_change", "autocorr_lag1", "change_frequency", "max_run_constant",
]

PAIR_FEATURE_COLS = [
    "corr_levels", "corr_changes", "frac_both_change",
    "frac_both_constant", "frac_same_direction",
    "ratio_cv", "mean_abs_diff",
]


# ---------------------------------------------------------------------------
#  Features
# ---------------------------------------------------------------------------

def extract_features(price_series):
    """Scale-invariant features from a 1D price time series. Works for any length >= 5."""
    p = np.asarray(price_series, dtype=float)
    p = p[~np.isnan(p)]
    if len(p) < 5:
        return None
    dp = np.diff(p)

    rng = p.max() - p.min()
    norm_level = (p.mean() - p.min()) / rng if rng > 1e-9 else 0.5

    autocorr_1 = np.corrcoef(p[:-1], p[1:])[0, 1] if p.std() > 1e-9 else 0.0
    autocorr_1 = 0.0 if np.isnan(autocorr_1) else autocorr_1

    change_threshold = 0.001 * (p.mean() or 1.0)
    change_freq = (np.abs(dp) > change_threshold).mean() if len(dp) else 0.0

    return {
        "norm_level":        norm_level,
        "coef_variation":    p.std() / p.mean() if p.mean() > 1e-9 else 0.0,
        "mean_abs_change":   np.abs(dp).mean() / p.mean() if p.mean() > 1e-9 else 0.0,
        "std_change":        dp.std() / p.mean() if p.mean() > 1e-9 else 0.0,
        "autocorr_lag1":     autocorr_1,
        "change_frequency":  change_freq,
        "max_run_constant":  _longest_constant_run(p) / len(p),
    }


def _longest_constant_run(p, tol=1e-9):
    """Longest run of consecutive identical (within tol) prices."""
    longest = current = 1
    for i in range(1, len(p)):
        if abs(p[i] - p[i-1]) < tol:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def extract_pair_features(price_a, price_b):
    """Cross-station features. Both arrays same length, >=5 obs after NaN removal."""
    a = np.asarray(price_a, dtype=float)
    b = np.asarray(price_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return None

    da, db = np.diff(a), np.diff(b)

    # Significant change threshold = 0.1% of local mean
    thresh_a, thresh_b = 0.001 * a.mean(), 0.001 * b.mean()
    changed_a = np.abs(da) > thresh_a
    changed_b = np.abs(db) > thresh_b

    # Level correlation
    corr_levels = (np.corrcoef(a, b)[0, 1]
                   if a.std() > 1e-9 and b.std() > 1e-9 else 0.0)

    # Change correlation -- the key coordination signal
    corr_changes = (np.corrcoef(da, db)[0, 1]
                    if da.std() > 1e-9 and db.std() > 1e-9 else 0.0)

    # Synchrony
    frac_both_change   = (changed_a & changed_b).mean()
    frac_both_constant = (~changed_a & ~changed_b).mean()

    both_idx = changed_a & changed_b
    frac_same_direction = (
        (np.sign(da[both_idx]) == np.sign(db[both_idx])).mean()
        if both_idx.sum() > 0 else 0.5
    )

    # Price gap
    ratio = a / b
    ratio_cv = ratio.std() / ratio.mean() if ratio.mean() > 0 else 0.0
    mean_abs_diff = np.abs(a - b).mean() / ((a.mean() + b.mean()) / 2)

    return {
        "corr_levels":          corr_levels if not np.isnan(corr_levels) else 0.0,
        "corr_changes":         corr_changes if not np.isnan(corr_changes) else 0.0,
        "frac_both_change":     frac_both_change,
        "frac_both_constant":   frac_both_constant,
        "frac_same_direction":  frac_same_direction,
        "ratio_cv":             ratio_cv,
        "mean_abs_diff":        mean_abs_diff,
    }


# ---------------------------------------------------------------------------
#  Training tables
# ---------------------------------------------------------------------------

def load_single(traj_dir=TRAJ_DIR):
    """One row per firm per run. `run` identifies the simulation the row came from."""
    idx = pd.read_csv(traj_dir / "index.csv")
    records = []
    for run_id, row in idx.iterrows():
        traj = np.load(traj_dir / row["traj_file"])   # shape (1000, 2)
        for firm_id in range(traj.shape[1]):
            feats = extract_features(traj[:, firm_id])
            if feats is None:
                continue
            feats.update({
                "delta": row["delta"], "alpha": row["alpha"],
                "beta": row["beta"], "seed": row["seed"],
                "firm": firm_id, "run": run_id,
                "setting": f"a{row['alpha']}_b{row['beta']}",
            })
            records.append(feats)
    df = pd.DataFrame(records)
    df["label"] = (df["delta"] > 0.5).astype(int)   # 1 = collusive
    return df


def load_pairs(traj_dir=TRAJ_DIR):
    """One row per run."""
    idx = pd.read_csv(traj_dir / "index.csv")
    records = []
    for run_id, row in idx.iterrows():
        traj = np.load(traj_dir / row["traj_file"])    # shape (1000, 2)
        feats = extract_pair_features(traj[:, 0], traj[:, 1])
        if feats is None:
            continue
        feats.update({
            "delta": row["delta"], "alpha": row["alpha"],
            "beta": row["beta"], "seed": row["seed"], "run": run_id,
            "setting": f"a{row['alpha']}_b{row['beta']}",
        })
        records.append(feats)
    df = pd.DataFrame(records)
    df["label"] = (df["delta"] > 0.5).astype(int)
    return df


# ---------------------------------------------------------------------------
#  Training and validation
# ---------------------------------------------------------------------------

def new_model():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)


def validate(df, cols):
    """Out-of-fold ROC AUC under three ways of splitting the same rows."""
    X = df[cols].values
    y = df["label"].values
    shuffled = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    by_run = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = [
        ("rows shuffled (leaks)",    shuffled,           None),
        ("whole runs held out",      by_run,             df["run"].values),
        ("whole settings held out",  LeaveOneGroupOut(), df["setting"].values),
    ]
    if len(df) == df["run"].nunique():      # one row per run: the first two are the same split
        splits = splits[1:]

    results = {}
    for name, cv, groups in splits:
        # Pooled over folds, because a held-out setting can contain one class only.
        proba = cross_val_predict(new_model(), X, y, cv=cv, groups=groups,
                                  method="predict_proba")[:, 1]
        line = f"  {name:<26} AUC {roc_auc_score(y, proba):.3f}"
        if not isinstance(cv, LeaveOneGroupOut):
            folds = cross_val_score(new_model(), X, y, cv=cv, groups=groups, scoring="roc_auc")
            line += f"   (mean of 5 folds {folds.mean():.3f} ± {folds.std():.3f})"
        print(line)
        results[name] = roc_auc_score(y, proba)
    return results


def plot_feature_histograms(df, cols, name):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, feat in zip(axes.flat, cols):
        for label, color in [(0, "tab:blue"), (1, "tab:orange")]:
            ax.hist(df.loc[df["label"] == label, feat],
                    bins=12, alpha=0.6, color=color,
                    label="collusive" if label == 1 else "competitive")
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, name)


def plot_roc_and_importance(clf, X, y, cols, name):
    """In-sample ROC and feature importance. The ROC here is not a test score."""
    y_score = clf.predict_proba(X)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y, y_score)
    axes[0].plot(fpr, tpr, lw=2, label=f"GBM, in-sample (AUC = {roc_auc_score(y, y_score):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC on simulated trajectories (in-sample)")
    axes[0].legend()

    importances = pd.Series(clf.feature_importances_, index=cols).sort_values()
    importances.plot.barh(ax=axes[1], color="steelblue")
    axes[1].set_xlabel("Feature importance")
    fig.tight_layout()
    _save(fig, name)


def _save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {FIG_DIR / name}")
    plt.close(fig)


def train(df, cols, out_file, tag):
    X = df[cols].values
    y = df["label"].values

    print(f"\n=== {tag} ===")
    print(f"Examples: {len(df)}  ({y.sum()} collusive, {len(df) - y.sum()} competitive) "
          f"from {df['run'].nunique()} runs in {df['setting'].nunique()} settings")
    print("Out-of-fold ROC AUC:")
    results = validate(df, cols)

    clf = new_model().fit(X, y)
    y_score = clf.predict_proba(X)[:, 1]
    print("\nConfusion matrix (in-sample):")
    print(pd.DataFrame(
        confusion_matrix(y, (y_score > 0.5).astype(int)),
        index=["actual competitive", "actual collusive"],
        columns=["pred competitive", "pred collusive"],
    ))

    with open(out_file, "wb") as f:
        pickle.dump({"model": clf, "feature_cols": cols}, f)
    print(f"Saved {out_file}")
    return clf, results


def main():
    if not (TRAJ_DIR / "index.csv").exists():
        raise SystemExit(f"No trajectories in {TRAJ_DIR}/. Run: python simulator.py trajectories")

    idx = pd.read_csv(TRAJ_DIR / "index.csv")
    on_threshold = ((idx["delta"] - 0.5).abs() < 1e-9).sum()
    if on_threshold:
        print(f"Note: {on_threshold} of {len(idx)} runs have delta equal to the 0.5 label "
              f"threshold to within rounding, so their label is arbitrary.")

    sim_df = load_single()
    plot_feature_histograms(sim_df, FEATURE_COLS, "features_single.png")
    clf, _ = train(sim_df, FEATURE_COLS, "trained_classifier.pkl", "Single-series classifier")
    plot_roc_and_importance(clf, sim_df[FEATURE_COLS].values, sim_df["label"].values,
                            FEATURE_COLS, "classifier_single.png")

    pair_df = load_pairs()
    plot_feature_histograms(pair_df, PAIR_FEATURE_COLS, "features_pair.png")
    train(pair_df, PAIR_FEATURE_COLS, "trained_classifier_pair.pkl", "Pair classifier")


if __name__ == "__main__":
    main()

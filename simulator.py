"""Q-learning agents in a repeated Bertrand pricing game with logit demand.

The model and the learning rule follow Calvano, Calzolari, Denicolo and
Pastorello (2020), "Artificial Intelligence, Algorithmic Pricing, and
Collusion", American Economic Review 110(10). The Model class and the
Q-learning loop are adapted from Matteo Courthoud's Python replication of that
paper (MIT licence, see NOTICE).

Each run is scored by its markup index

    delta = (average price - Nash price) / (monopoly price - Nash price)

so 0 is the competitive outcome and 1 is joint monopoly.

Usage
    python simulator.py smoke          one run, about 15 seconds
    python simulator.py sweep          3 x 3 grid of learning rate and exploration decay
    python simulator.py trajectories   48 labelled price paths for the classifier
"""

import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

np.set_printoptions(precision=3, suppress=True)

FIG_DIR = "figures"

# Shorter learning runs than the paper, so a laptop can finish the experiments.
TSTABLE = 3e4      # periods the greedy policy must stay unchanged to call convergence
TMAX = 5e5         # give up after this many periods


# ---------------------------------------------------------------------------
#  The game
# ---------------------------------------------------------------------------

class Model:
    """Bertrand n-opoly with logit demand and Q-learning agents."""

    def __init__(self, **kwargs):
        self.n = kwargs.get("n", 2)                 # firms
        self.alpha = kwargs.get("alpha", 0.15)      # learning rate
        self.beta = kwargs.get("beta", 4e-6)        # exploration decay
        self.delta = kwargs.get("delta", 0.95)      # discount factor
        self.c = kwargs.get("c", 1.0)               # marginal cost
        self.a = kwargs.get("a", 2.0)               # product quality
        self.a0 = kwargs.get("a0", 0.0)             # outside good
        self.mu = kwargs.get("mu", 0.25)            # horizontal differentiation
        self.k = kwargs.get("k", 15)                # prices on the grid
        self.tstable = int(kwargs.get("tstable", 1e5))
        self.tmax = int(kwargs.get("tmax", 1e7))

        self.sdim = (self.k,) * self.n
        self.s0 = np.zeros(self.n, dtype=int)
        self.p_competitive, self.p_monopoly = self._benchmarks()
        self.A = self._action_grid()
        self.PI = self._profit_table()
        self.Q = self._init_Q()

    def demand(self, p):
        e = np.exp((self.a - p) / self.mu)
        return e / (np.sum(e) + np.exp(self.a0 / self.mu))

    def _foc_nash(self, p):
        d = self.demand(p)
        return np.squeeze(1 - (p - self.c) * (1 - d) / self.mu)

    def _foc_monopoly(self, p):
        d = self.demand(p); d1, p1 = np.flip(d), np.flip(p)
        return np.squeeze(1 - (p - self.c) * (1 - d) / self.mu
                          + (p1 - self.c) * d1 / self.mu)

    def _benchmarks(self):
        p0 = np.ones((1, self.n)) * 3 * self.c
        return fsolve(self._foc_nash, p0), fsolve(self._foc_monopoly, p0)

    def _action_grid(self):
        a = np.linspace(min(self.p_competitive), max(self.p_monopoly), self.k - 2)
        d = a[1] - a[0]
        return np.linspace(min(a) - d, max(a) + d, self.k)

    def _profit_table(self):
        PI = np.zeros(self.sdim + (self.n,))
        for s in product(*[range(i) for i in self.sdim]):
            p = np.asarray(self.A[np.asarray(s)])
            PI[s] = (p - self.c) * self.demand(p)
        return PI

    def _init_Q(self):
        Q = np.zeros((self.n,) + self.sdim + (self.k,))
        for n in range(self.n):
            pi = np.mean(self.PI[..., n], axis=tuple(i for i in range(self.n) if i != n))
            Q[n] = np.tile(pi, self.sdim + (1,)) / (1 - self.delta)
        return Q


# ---------------------------------------------------------------------------
#  Q-learning
# ---------------------------------------------------------------------------

def _pick_actions(game, s, t):
    a = np.zeros(game.n, dtype=int)
    pr_explore = np.exp(-t * game.beta)
    explore = pr_explore > np.random.rand(game.n)
    for n in range(game.n):
        if explore[n]:
            a[n] = np.random.randint(0, game.k)
        else:
            a[n] = np.argmax(game.Q[(n,) + tuple(s)])
    return a


def _update_q(game, s, a, s1, pi, stable):
    for n in range(game.n):
        subj = (n,) + tuple(s) + (a[n],)
        old = game.Q[subj]
        new = pi[n] + game.delta * np.max(game.Q[(n,) + tuple(s1)])
        old_argmax = np.argmax(game.Q[(n,) + tuple(s)])
        game.Q[subj] = (1 - game.alpha) * old + game.alpha * new
        new_argmax = np.argmax(game.Q[(n,) + tuple(s)])
        stable = (stable + (old_argmax == new_argmax)) * (old_argmax == new_argmax)
    return stable


def _simulate(game):
    s = game.s0.copy(); stable = 0
    for t in range(game.tmax):
        a = _pick_actions(game, s, t)
        stable = _update_q(game, s, a, a, game.PI[tuple(a)], stable)
        s = a
        if stable > game.tstable:
            return True, t, s
    return False, game.tmax, s


def _greedy_trajectory(game, s_init, n_periods=1000):
    s = s_init.copy(); prices = []
    for _ in range(n_periods):
        a = np.array([np.argmax(game.Q[(n,) + tuple(s)]) for n in range(game.n)])
        prices.append(game.A[a]); s = a
    return np.array(prices)


def run_one(seed=0, capture_trajectory=False, **kwargs):
    """Learn until convergence, then play the greedy policy for 1000 periods."""
    np.random.seed(seed)
    game = Model(**kwargs)
    converged, t_final, s_final = _simulate(game)
    traj = _greedy_trajectory(game, s_final, n_periods=1000)
    p_nash = float(game.p_competitive[0])
    p_mono = float(game.p_monopoly[0])
    p_avg = float(traj.mean())
    result = {
        "seed": seed, "alpha": game.alpha, "beta": game.beta, "n": game.n,
        "converged": converged, "t_final": t_final,
        "p_nash": p_nash, "p_monopoly": p_mono, "p_avg": p_avg,
        "delta": (p_avg - p_nash) / (p_mono - p_nash),
    }
    if capture_trajectory:
        result["trajectory"] = traj
    return result


def _greedy_trajectory_perturbed(game, s_init, n_periods=1000, perturb_prob=0.02, rng=None):
    """Greedy policy with small per-period chance of a random deviation."""
    if rng is None:
        rng = np.random
    s = s_init.copy()
    prices = []
    for _ in range(n_periods):
        a = np.zeros(game.n, dtype=int)
        for i in range(game.n):
            if rng.rand() < perturb_prob:
                a[i] = rng.randint(0, game.k)
            else:
                a[i] = np.argmax(game.Q[(i,) + tuple(s)])
        prices.append(game.A[a])
        s = a
    return np.array(prices)


def run_one_perturbed(seed=0, perturb_prob=0.02, **kwargs):
    """Like run_one, but the saved trajectory is from perturbed greedy play.

    A converged pair of agents sits on one flat price, which says nothing about
    how they respond to each other. Forcing an occasional random price makes the
    punishment and recovery visible. Delta is still measured from pure greedy
    play, so the label means the same thing in both kinds of run.
    """
    np.random.seed(seed)
    game = Model(**kwargs)
    converged, t_final, s_final = _simulate(game)

    traj_pure = _greedy_trajectory(game, s_final, n_periods=1000)
    p_nash = float(game.p_competitive[0])
    p_mono = float(game.p_monopoly[0])
    p_avg_pure = float(traj_pure.mean())
    delta = (p_avg_pure - p_nash) / (p_mono - p_nash)

    pert_rng = np.random.RandomState(seed + 99999)
    traj_pert = _greedy_trajectory_perturbed(
        game, s_final, n_periods=1000, perturb_prob=perturb_prob, rng=pert_rng,
    )

    return {
        "seed": seed, "alpha": game.alpha, "beta": game.beta, "n": game.n,
        "converged": converged, "t_final": t_final,
        "p_nash": p_nash, "p_monopoly": p_mono, "p_avg_pure": p_avg_pure,
        "delta": delta,
        "trajectory": traj_pert,
    }


# ---------------------------------------------------------------------------
#  Experiments
# ---------------------------------------------------------------------------

def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {path}")
    plt.close(fig)


def smoke_test():
    """One run at the baseline parameters."""
    t0 = time.time()
    result = run_one(seed=0, capture_trajectory=True, tstable=TSTABLE, tmax=TMAX)
    print(f"Time: {time.time() - t0:.1f}s")
    print(f"Converged: {result['converged']} at t={result['t_final']:,}")
    print(f"Nash:     {result['p_nash']:.3f}")
    print(f"Monopoly: {result['p_monopoly']:.3f}")
    print(f"Average:  {result['p_avg']:.3f}")
    print(f"Delta:    {result['delta']:.3f}")

    traj = result["trajectory"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(traj[:200, 0], label="Firm 1", lw=1.5)
    ax.plot(traj[:200, 1], label="Firm 2", lw=1.5, alpha=0.8)
    ax.axhline(result["p_nash"], color="gray", ls="--", label="Nash")
    ax.axhline(result["p_monopoly"], color="black", ls="--", label="Monopoly")
    ax.set_xlabel("Period (greedy play after convergence)")
    ax.set_ylabel("Price")
    ax.set_title(f"Converged price trajectory  (Delta = {result['delta']:.2f})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "smoke_trajectory.png")


SWEEP_ALPHAS = [0.05, 0.15, 0.25]
SWEEP_BETAS = [1e-6, 1e-5, 5e-5]
SWEEP_SEEDS = 3


def sweep():
    """Markup index across learning rate and exploration decay."""
    records = []
    total = len(SWEEP_ALPHAS) * len(SWEEP_BETAS) * SWEEP_SEEDS
    t_start = time.time()
    i = 0
    for alpha in SWEEP_ALPHAS:
        for beta in SWEEP_BETAS:
            for seed in range(SWEEP_SEEDS):
                i += 1
                t0 = time.time()
                r = run_one(seed=seed, alpha=alpha, beta=beta,
                            tstable=TSTABLE, tmax=TMAX)
                records.append(r)
                print(f"  [{i:>2}/{total}] alpha={alpha:.2f} beta={beta:.0e} seed={seed}  "
                      f"Delta={r['delta']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nTotal: {time.time() - t_start:.0f}s")
    df = pd.DataFrame(records)
    df.to_csv("mini_sweep_results.csv", index=False)
    print("Wrote mini_sweep_results.csv")
    plot_heatmap(df)
    return df


def plot_heatmap(df):
    pivot = (df.groupby(["alpha", "beta"])
               .delta.mean()
               .unstack("beta"))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{b:.0e}" for b in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{a:.2f}" for a in pivot.index])
    ax.set_xlabel("Exploration decay beta")
    ax.set_ylabel("Learning rate alpha")
    ax.set_title("Mean Delta (markup index) across seeds")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if pivot.values[i, j] > 0.5 else "black",
                    fontsize=11)

    fig.colorbar(im, ax=ax, label="Delta  (0 = Nash, 1 = Monopoly)")
    fig.tight_layout()
    _save(fig, "fig1_heatmap.png")


# Hand-picked cells: a mix of likely-collusive and likely-competitive parameter
# regions, so the classifier has both classes to learn from.
TRAJECTORY_CONFIGS = [
    {"alpha": 0.25, "beta": 1e-6},   # likely collusive
    {"alpha": 0.20, "beta": 5e-6},   # likely collusive
    {"alpha": 0.15, "beta": 1e-5},   # middle
    {"alpha": 0.10, "beta": 1e-5},   # middle
    {"alpha": 0.05, "beta": 5e-5},   # likely competitive
    {"alpha": 0.05, "beta": 1e-4},   # likely competitive
]
TRAJECTORY_SEEDS = 8          # 6 cells x 8 seeds = 48 trajectories
PERTURB_PROB = 0.02


def make_trajectories(perturbed=True):
    """Save one labelled 1000-period price path per run.

    perturbed=True writes trajectories_perturbed/, the set the classifier is
    trained on. perturbed=False writes trajectories/, pure greedy play.
    """
    out_dir = "trajectories_perturbed" if perturbed else "trajectories"
    os.makedirs(out_dir, exist_ok=True)

    records = []
    t_start = time.time()
    total = len(TRAJECTORY_CONFIGS) * TRAJECTORY_SEEDS
    i = 0
    for cfg in TRAJECTORY_CONFIGS:
        for seed in range(TRAJECTORY_SEEDS):
            i += 1
            t0 = time.time()
            if perturbed:
                r = run_one_perturbed(seed=seed, perturb_prob=PERTURB_PROB,
                                      tstable=TSTABLE, tmax=TMAX, **cfg)
            else:
                r = run_one(seed=seed, capture_trajectory=True,
                            tstable=TSTABLE, tmax=TMAX, **cfg)
            fname = f"traj_a{cfg['alpha']:.3f}_b{cfg['beta']:.0e}_s{seed}.npy"
            np.save(os.path.join(out_dir, fname), r.pop("trajectory"))
            r["traj_file"] = fname
            records.append(r)
            print(f"  [{i:>2}/{total}] a={cfg['alpha']:.2f} b={cfg['beta']:.0e} seed={seed}  "
                  f"Delta={r['delta']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(records).to_csv(os.path.join(out_dir, "index.csv"), index=False)
    print(f"\nDone in {time.time() - t_start:.0f}s. Wrote {len(records)} trajectories to {out_dir}/")
    plot_trajectories(out_dir)


def plot_trajectories(traj_dir="trajectories_perturbed"):
    """The most and the least collusive run, side by side."""
    idx = pd.read_csv(os.path.join(traj_dir, "index.csv"))
    high_d = idx.sort_values("delta", ascending=False).iloc[0]
    low_d = idx.sort_values("delta", ascending=True).iloc[0]

    t_high = np.load(os.path.join(traj_dir, high_d["traj_file"]))
    t_low = np.load(os.path.join(traj_dir, low_d["traj_file"]))
    kind = "Perturbed" if "perturbed" in traj_dir else "Greedy"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, t, row, title in [
        (axes[0], t_high, high_d, f"{kind} collusive  (Delta = {high_d['delta']:.2f})"),
        (axes[1], t_low,  low_d,  f"{kind} competitive  (Delta = {low_d['delta']:.2f})"),
    ]:
        ax.plot(t[:300, 0], lw=1.0, label="Firm 1")
        ax.plot(t[:300, 1], lw=1.0, alpha=0.8, label="Firm 2")
        ax.axhline(row["p_nash"], color="gray", ls="--", label="Nash")
        ax.axhline(row["p_monopoly"], color="black", ls="--", label="Monopoly")
        ax.set_xlabel("Period"); ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("Price")
    fig.tight_layout()
    _save(fig, "fig2_trajectories.png" if "perturbed" in traj_dir else "trajectories_greedy.png")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if command == "smoke":
        smoke_test()
    elif command == "sweep":
        sweep()
    elif command == "trajectories":
        make_trajectories(perturbed="--greedy" not in sys.argv)
    else:
        sys.exit(__doc__)

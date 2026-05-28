"""
HMM regime model for daily stock returns -- starter code.

Pipeline:
  1. Download prices (yfinance) -> daily log returns
  2. Chronological train / test split
  3. Fit Gaussian HMMs with K regimes (several restarts; EM hits local optima)
  4. Sweep K: held-out log-likelihood + BIC  (the "how does performance change" axis)
  5. Decode regimes (Viterbi) and shade them under the price curve (interpretation)

Run:   python hmm_starter.py
Deps:  pip install yfinance hmmlearn numpy pandas matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# ----------------------------------------------------------------- config
TICKER = "AAPL"
START, END = "2005-01-01", "2024-01-01"
TRAIN_FRAC = 0.8
K_VALUES = [1, 2, 3, 4, 5]
N_RESTARTS = 5  # keep the best of several fits (EM is sensitive to init)
SEED = 0
np.random.seed(SEED)


# ----------------------------------------------------------------- data
def load_returns(ticker=TICKER, start=START, end=END):
    """Download close prices and return (log-returns as (N,1) array, aligned price Series)."""
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)[
        "Close"
    ]
    if isinstance(px, pd.DataFrame):  # some yfinance versions return a 1-col frame
        px = px.iloc[:, 0]
    px = px.dropna()
    rets = np.log(px / px.shift(1)).dropna()
    return rets.values.reshape(-1, 1), px.loc[rets.index]


def split(X, frac=TRAIN_FRAC):
    """Chronological split -- never shuffle a time series."""
    n = int(frac * len(X))
    return X[:n], X[n:], n


# ----------------------------------------------------------------- model
def fit_hmm(X, k, n_restarts=N_RESTARTS):
    """Fit a K-state Gaussian HMM, keeping the restart with the best training log-likelihood."""
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        m = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=1000,
            tol=1e-4,
            random_state=SEED + r,
        )
        try:
            m.fit(X)
            ll = m.score(X)
        except Exception:
            continue  # a bad init can fail to converge; just skip it
        if ll > best_ll:
            best, best_ll = m, ll
    return best


def n_params(k):
    """Free parameters of a 1-D diagonal Gaussian HMM with K states (used for BIC)."""
    return (k - 1) + k * (k - 1) + k + k  # startprob + transitions + means + variances


def bic(model, X):
    return -2 * model.score(X) + n_params(model.n_components) * np.log(len(X))


# ----------------------------------------------------------------- experiments
def sweep_k(train, test):
    """Fit each K and record train/held-out log-likelihood per day and BIC."""
    rows = []
    for k in K_VALUES:
        m = fit_hmm(train, k)
        if m is None:
            continue
        rows.append(
            {
                "K": k,
                "train_LL_per_day": m.score(train) / len(train),
                "test_LL_per_day": m.score(test)
                / len(test),  # held-out predictive loss
                "BIC": bic(m, train),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- interpretation
def plot_regimes(price, X, model, fname="regimes.png"):
    """Viterbi-decode the regimes and shade contiguous runs under the price curve."""
    states = model.predict(X)
    # relabel states by volatility so colors are consistent (0 = calmest regime)
    order = np.argsort(model.covars_.flatten())
    relabel = {old: new for new, old in enumerate(order)}
    states = np.array([relabel[s] for s in states])

    colors = plt.cm.coolwarm(np.linspace(0, 1, model.n_components))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(price.index, price.values, color="black", lw=0.8, zorder=3)

    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            ax.axvspan(
                price.index[start],
                price.index[i - 1],
                color=colors[states[start]],
                alpha=0.25,
                zorder=1,
            )
            start = i

    ax.set_title(
        f"{TICKER}: price with {model.n_components} decoded regimes (red = high volatility)"
    )
    ax.set_ylabel("price")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"saved {fname}")


# ----------------------------------------------------------------- main
def main():
    X, price = load_returns()
    train, test, _ = split(X)
    print(f"{len(X)} days  ->  train {len(train)}  /  test {len(test)}\n")

    results = sweep_k(train, test)
    print(results.to_string(index=False))

    best_k = int(results.loc[results["test_LL_per_day"].idxmax(), "K"])
    print(f"\nbest K by held-out log-likelihood: {best_k}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results["K"], results["test_LL_per_day"], "o-")
    ax.set_xlabel("number of regimes K")
    ax.set_ylabel("held-out log-likelihood / day")
    ax.set_title("Model selection: held-out fit vs K")
    fig.tight_layout()
    fig.savefig("k_sweep.png", dpi=120)
    print("saved k_sweep.png")

    final_model = fit_hmm(
        train, max(best_k, 2)
    )  # use >=2 states for the regime picture

    plot_regimes(price, X, final_model)


def k_1_model():
    X, price = load_returns()
    train, test, _ = split(X)
    print(f"{len(X)} days  ->  train {len(train)}  /  test {len(test)}\n")

    results = sweep_k(train, test)
    print(results.to_string(index=False))

    best_k = int(results.loc[results["test_LL_per_day"].idxmax(), "K"])
    print(f"\nbest K by held-out log-likelihood: {best_k}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results["K"], results["test_LL_per_day"], "o-")
    ax.set_xlabel("number of regimes K")
    ax.set_ylabel("held-out log-likelihood / day")
    ax.set_title("Model selection: held-out fit vs K")
    fig.tight_layout()
    fig.savefig("k_sweep.png", dpi=120)
    print("saved k_sweep.png")

    final_model = fit_hmm(train, 3)  # use >=2 states for the regime picture

    plot_regimes(price, X, final_model)


if __name__ == "__main__":
    k_1_model()

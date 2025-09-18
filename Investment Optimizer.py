import time
import numpy as np, pandas as pd, yfinance as yf, cvxpy as cp
import matplotlib.pyplot as plt

TRADING_DAYS = 252

# q to quit
def getInput(prompt, default=None, cast=str):
    val = input(prompt).strip()
    if val.lower() == "q":
        print("Exiting...")
        raise SystemExit
    if val == "" and default is not None:
        return default
    return cast(val)

# welcome message
def welcome():
    print("\nPortfolio Optimization Tool")
    print("------------------------------------------------------------")
    print("This program applies Modern Portfolio Theory to help you")
    print("analyze how to allocate investments across a set of stocks.")
    print("It compares two approaches:")
    print(" • Continuous optimization (Markowitz model, long-only).")
    print(" • Discrete k-equal portfolios (exactly k stocks, equal weights).\n")

    print("How it works:")
    print(" - Retrieves historical stock prices from Yahoo Finance.")
    print(" - Estimates expected returns and the covariance matrix.")
    print(" - Balances expected return against risk (variance), controlled by λ.")
    print(" - Reports portfolio return, variance, Sharpe ratio, and allocations.")
    print(" - Produces a bar chart comparing the two allocation methods.\n")
    print(" - Click q to exit program anytime.\n")

    print("Usage instructions:")
    print(" - You will be asked how many companies to include and their tickers.")
    print(" - Input tickers separated by spaces or commas (e.g., AAPL MSFT NVDA).")
    print(" - Dates, λ (risk aversion), risk-free rate, and k can be adjusted.")
    print(" - At any prompt, type 'q' to quit immediately.\n")

    print("Examples of commonly used tickers you can evaluate:")
    print("   • Technology: AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA), GOOGL (Alphabet), AMZN (Amazon), META (Meta).")
    print("   • Finance: JPM (JPMorgan), GS (Goldman Sachs), BAC (Bank of America).")
    print("   • Energy: XOM (ExxonMobil), CVX (Chevron).")
    print("   • Other: TSLA (Tesla), NFLX (Netflix), DIS (Disney).\n")

    print("Example run:")
    print("   Companies: 3 → AAPL AMZN JPM")
    print("   Start date: 2022-01-01, End date: 2025-09-01")
    print("   λ: 0.5, Risk-free: 0.02, k: 3")
    print("   The program will then compute allocations, risk/return trade-offs,")
    print("   and display results in text and a comparison chart.\n")
    print("------------------------------------------------------------\n")

# helpers
def loadPrices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all").dropna(axis=1, how="any")

def logReturns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def statsAnnualized(prices):
    r = logReturns(prices)
    mu = r.mean().values * TRADING_DAYS
    Sig = np.cov(r.values.T) * TRADING_DAYS
    return mu, Sig, len(r)

def mvObjective(w, mu, Sig, lam):
    er  = float(np.dot(mu, w))
    var = float(np.dot(w, np.dot(Sig, w)))
    return er - lam * var, er, var

def sharpe(er, var, rf):
    sd = max(np.sqrt(max(var, 0.0)), 1e-12)
    return (er - rf) / sd

# solvers
def solveContinuous(mu, Sig, lam, rf):
    n = len(mu)
    w = cp.Variable(n, nonneg=True)
    mu_vec = np.asarray(mu, dtype=float)
    Sig_mat = np.asarray(Sig, dtype=float)
    objective = cp.Minimize(lam * cp.quad_form(w, Sig_mat) - cp.sum(cp.multiply(mu_vec, w)))
    constraints = [cp.sum(w) == 1]
    prob = cp.Problem(objective, constraints)

    t0 = time.time()
    solved = False
    for solver in (cp.ECOS, cp.OSQP, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                solved = True
                break
        except Exception:
            continue
    t = time.time() - t0
    if not solved:
        raise RuntimeError("continuous solver failed")

    w = np.asarray(w.value, dtype=float).ravel()
    obj, er, var = mvObjective(w, mu, Sig, lam)
    return {"w": w, "obj": obj, "er": er, "var": var, "sharpe": sharpe(er, var, rf), "time": t}

def solveDiscrete(mu, Sig, lam, k, rf):
    n = len(mu)
    k = max(1, min(k, n))
    chosen, pool = [], list(range(n))
    t0 = time.time()
    for _ in range(k):
        best_val, best_i = None, None
        for i in pool:
            x = np.zeros(n, dtype=int)
            if chosen: x[chosen] = 1
            x[i] = 1
            val = (lam/(k*k)) * float(np.dot(x, np.dot(Sig, x))) - (1.0/k) * float(np.dot(mu, x))
            if (best_val is None) or (val < best_val):
                best_val, best_i = val, i
        chosen.append(best_i); pool.remove(best_i)
    t = time.time() - t0

    x = np.zeros(n, dtype=int); x[chosen] = 1
    w = x.astype(float) / k
    obj, er, var = mvObjective(w, mu, Sig, lam)
    return {"w": w, "picked_idx": chosen, "obj": obj, "er": er, "var": var,
            "sharpe": sharpe(er, var, rf), "time": t}

# user input flow
def askTickers():
    n = getInput("How many companies? (default 5) > ", default=5, cast=int)
    raw = getInput("Enter tickers (AAPL MSFT NVDA ... or q to quit): ",
                    default="AAPL MSFT NVDA AMZN GOOGL")
    tickers = raw.replace(",", " ").split()
    tickers = [t.upper() for t in tickers][:n]
    if len(tickers) == 0:
        tickers = ["AAPL", "MSFT", "NVDA"]
    return tickers

def main():
    welcome()
    tickers = askTickers()
    start = getInput("Start date [YYYY-MM-DD] (default 2022-01-01): ", default="2022-01-01")
    end = getInput("End date   [YYYY-MM-DD] (default 2025-09-01): ", default="2025-09-01")
    lam = getInput("Risk aversion λ (default 0.5): ", default=0.5, cast=float)
    rf = getInput("Risk-free (annual) (default 0.02): ", default=0.02, cast=float)
    k = getInput("k for k-equal discrete (default 3): ", default=3, cast=int)

    print("\nDownloading prices...")
    px = loadPrices(tickers, start, end)
    used = list(px.columns)
    if px.empty or len(used) < 2:
        raise SystemExit("Need at least two tickers with complete data.")

    mu, Sig, _ = statsAnnualized(px)

    cont = solveContinuous(mu, Sig, lam, rf)
    disc = solveDiscrete(mu, Sig, lam, k, rf)
    disc["picked"] = [used[i] for i in disc["picked_idx"]]

    # report
    print("\nContinuous (long-only)")
    print(f"time: {cont['time']:.3f}s")
    print(f"obj: {cont['obj']:.6f}")
    print(f"return: {cont['er']:.4f} | var: {cont['var']:.6f} | sharpe: {cont['sharpe']:.4f}")
    print(f"weights: {np.round(cont['w'],4)} (sum={cont['w'].sum():.4f})")

    print(f"\nDiscrete k-equal (k={min(k, len(used))}, greedy)")
    print(f"time: {disc['time']:.3f}s")
    print(f"obj: {disc['obj']:.6f}")
    print(f"return: {disc['er']:.4f} | var: {disc['var']:.6f} | sharpe: {disc['sharpe']:.4f}")
    print(f"picked: {disc['picked']} (idx {disc['picked_idx']})")
    print(f"weights: {np.round(disc['w'],4)}")

    # quick plot
    pos = np.arange(len(used))
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(pos-0.25, cont['w'], width=0.5, label="Continuous")
    ax.bar(pos+0.25, disc['w'], width=0.5, label=f"k-equal (k={min(k, len(used))}, greedy)")
    ax.set_xticks(pos); ax.set_xticklabels(used, rotation=45, ha="right")
    ax.set_ylabel("Weight"); ax.set_title("Portfolio Weights: Continuous vs k-Equal")
    ax.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()

# Quant Portfolio

A collection of projects exploring **portfolio optimization** and **quantitative finance** techniques.  
The goal is to implement practical tools for investment allocation, risk management, and performance evaluation. 

---

## Projects

### 1. Investment Optimizer
Interactive script that applies **Modern Portfolio Theory (MPT)** to optimize stock allocations.

- **Features**
  - Continuous Markowitz optimization (long-only, convex optimization with CVXPY).
  - Discrete *k*-equal allocation (greedy selection of exactly *k* stocks, equal weights).
  - Sharpe ratio, expected return, and variance metrics.
  - Simple bar chart comparing allocations.
  - Interactive input (choose tickers, time period, risk aversion λ, risk-free rate, etc.).
  - Exit any time with `q`.

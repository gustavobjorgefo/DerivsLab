# DerivsLab – Option Pricing and Volatility Research

This repository contains code, models, and utilities for option pricing, volatility modeling, and quantitative research in derivatives.

The project serves as a personal laboratory where I explore both theoretical and practical aspects of derivatives, from closed-form solutions like Black-Scholes to numerical methods such as Monte Carlo and finite differences. It also includes tools for implied volatility extraction from market data and the infrastructure needed to fetch and process such data.   

---

## 📚 About DerivsLab

The aim of this project is to build a flexible environment for experimenting with:

- Option pricing models — closed-form solutions and numerical methods
- Volatility modeling — historical, realized, and stochastic approaches
- Implied volatility calculation — extracting IV from market data
- Greeks and sensitivities — risk measures and scenario analysis
- Exotic options — barrier, Asian, lookback, and other structures
- Data infrastructure — utilities for requesting and cleaning market data

The philosophy is to connect theory to implementation, with transparent and reproducible code that can be extended for both educational and research purposes.

---

## 📁 Repository Structure

Planned modules and structure (to evolve as the project grows):

| Folder / Module | Description |
|--------|-------------|
| pricing/ | Closed-form models (e.g. Black-Scholes, Bachelier) and numerical pricers (binomial, trinomial, Monte Carlo). |
| volatility/ | Historical and implied volatility calculations, stochastic volatility models. |
| greeks/ | Sensitivities and risk measures for vanilla and exotic options. |
| exotics/ | Pricing routines for barrier, Asian, lookback, and other exotic payoffs. |
| data/ | Data access layer: connectors, request utilities, and preprocessing. |
| notebooks/ | Example notebooks with use cases and demonstrations. |

---

## 🧰 Tools and Dependencies

The code is primarily written in **Python**, and makes use of libraries such as:

- `pandas`, `numpy` — data handling and numerical computing
- `matplotlib`, `seaborn` — visualization
- `scipy`, `statsmodels`, etc. — statistical tools where needed
- `requests`, `yfinance`, or APIs — for market data retrieval

Some notebooks may require additional dependencies listed in a `requirements.txt`.

---

## 🌐 Background & Motivation

DerivsLab is not just a pricing library — it is a research environment where I document and test methods at the intersection of:

- Derivatives theory
- Numerical methods
- Volatility and risk modeling
- Data engineering for quantitative finance

The repository is meant to remain open and extensible, so that each module or notebook can serve as a reference point for future work.

---

## 📜 Disclaimer

This repository is for educational and research purposes only.  
It does **not** constitute financial advice or a recommendation to trade.  
Always conduct your own due diligence and consult a qualified advisor when making investment decisions.

---

## 🤝 Contributions & Feedback

This is a personal learning project, but I'm always open to discussion, suggestions, or improvements. Feel free to fork, open issues, or reach out!
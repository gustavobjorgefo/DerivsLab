# DerivsLab

**Quantitative research and simulation environment for derivatives pricing, volatility modeling, and portfolio-level risk.**

---

## Overview

DerivsLab is a research environment for derivatives pricing, volatility modeling, and quantitative risk analysis. It connects theoretical models — closed-form solutions, numerical methods, and stochastic processes — to working, testable implementations.

The project is structured around three areas of focus:

- **Pricing and volatility research.** Closed-form and numerical pricing models, historical and implied volatility estimation, and stochastic volatility frameworks.
- **Instruments and risk measures.** A consistent representation of derivatives contracts (vanilla and exotic) and their associated sensitivities (Greeks).
- **Book-level simulation.** An emerging layer for modeling a derivatives portfolio over time — positions, hedging strategies, and daily mark-to-market — under both historical and simulated market scenarios.

The intent is for theory, implementation, and simulation to sit side by side: every model is meant to be tested not only in isolation, but also in the context of a position or strategy evolving day by day.

---

## Scope and Direction

DerivsLab is under active, incremental development. The table below reflects the current state of each area, distinguishing what is implemented from what is in progress or planned.

| Area | Status | Description |
|---|---|---|
| Pricing | Implemented | Closed-form models (Black-Scholes) and Monte Carlo pricing under GBM, including discretely monitored barrier options. |
| Volatility | Implemented | Historical and EWMA estimators, GARCH modeling, implied volatility extraction, and SVI parametrization. |
| Greeks | Implemented | Sensitivities for vanilla options. |
| Instruments | In progress | Typed representations for option, equity, bond, and future contracts, including barrier-option structures. |
| Simulation | In progress | Path generation and Monte Carlo simulation engines, forming the basis for scenario-driven, day-by-day repricing. |
| Risk | Planned | Portfolio-level aggregation of Greeks, P&L decomposition, and scenario/stress analysis across a book of positions. |
| Data | Implemented | Market data retrieval and local caching utilities. |

The longer-term direction is to support a book-level simulation flow: a position is opened (for example, an option sold at a given implied volatility), a strategy governs how it is hedged over time (for example, a daily delta hedge), and the book is repriced and marked to market under a scenario engine that can run on historical or simulated data. This is designed to scale to multiple positions, instruments, and strategies running concurrently within the same book.

---

## Repository Structure

```
derivslab/
├── pricing/        Closed-form and numerical pricing models
├── volatility/      Historical, EWMA, GARCH, implied volatility, SVI
├── greeks/          Sensitivities for vanilla and exotic instruments
├── instruments/     Contract definitions: option, equity, bond, future
├── simulation/       Path generation and Monte Carlo engines
├── risk/            Portfolio-level risk and P&L analysis (planned)
└── utils/           Data retrieval and caching utilities
```

This reflects the current package layout under `src/derivslab/`. It will evolve as the book-level simulation layer (`risk/` and related modules) matures.

---

## Getting Started

### Requirements

- Python 3.8 or later
- Git

### Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/<your-username>/derivslab.git
cd derivslab

python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

For development (testing and linting):

```bash
pip install -r requirements-dev.txt
```

### Running Tests

```bash
pytest
```

---

## Tools and Dependencies

The project is written in Python and relies on:

- `numpy`, `pandas` — numerical computing and data handling
- `scipy`, `statsmodels`, `sympy` — statistical and symbolic methods
- `matplotlib` — visualization
- `yfinance` — market data retrieval

Development dependencies (`requirements-dev.txt`) include `pytest`, `black`, `isort`, `flake8`, and `pre-commit` for testing and code style enforcement.

---

## Disclaimer

This repository is for research and educational purposes only. It does not constitute financial advice or a recommendation to trade. Always conduct independent due diligence and consult a qualified advisor before making investment decisions.

---

## Status

DerivsLab is an independent, evolving project under active development. The architecture and scope described above represent the current direction rather than a finished system. Feedback, questions, and discussion are welcome — feel free to open an issue or reach out directly.
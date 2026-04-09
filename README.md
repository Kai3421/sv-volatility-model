# Bayesian Analysis of Stochastic Volatility Models for Stock Returns

Bachelor's thesis by Kai Koepchen, University of Cologne, 2023.

This project estimates a Stochastic Volatility (SV) model for equity log returns using Bayesian inference via Markov Chain Monte Carlo methods. The latent log-volatility path is sampled through a Griddy-Gibbs data augmentation scheme, while model parameters are drawn from their conjugate conditional posteriors. The SV model is benchmarked against a GARCH(1,1) baseline using Value-at-Risk coverage tests (Kupiec unconditional coverage and Christoffersen independence/conditional coverage). The empirical application uses daily Berkshire Hathaway Class B (BRK-B) stock data from 2018 to 2023.

## Method Overview

**Stochastic Volatility Model:**

$$r_t = \sqrt{h_t} \, \varepsilon_t, \quad \varepsilon_t \sim N(0,1)$$

$$\ln h_t = \alpha_0 + \alpha_1 \ln h_{t-1} + v_t, \quad v_t \sim N(0, \sigma_v^2)$$

**Bayesian Inference:**
- Latent volatilities $h_t$ are treated as missing data and sampled via the **Griddy-Gibbs sampler** (Ritter & Tanner, 1992), which evaluates the unnormalized conditional posterior on a fine grid and draws via the inverse CDF method.
- Parameters $(\alpha_0, \alpha_1)$ are drawn from their **conjugate normal posterior** given the log-volatility path.
- $\sigma_v^2$ is drawn from its **conjugate inverse-chi-squared posterior**.
- Initial volatilities are seeded from a GARCH(1,1) fit to accelerate convergence.

**VaR Evaluation:**
- 5% Value-at-Risk computed as $\text{VaR}_t(\alpha) = \Phi^{-1}(\alpha) \cdot \sqrt{h_t}$
- **Kupiec (1995)** unconditional coverage test (proportion of failures)
- **Christoffersen (1998)** independence test (serial correlation of violations)
- Conditional coverage test (combined)

## Results Summary

Both the SV model and GARCH(1,1) pass the Kupiec and Christoffersen tests at the 5% significance level on the BRK-B dataset. The SV model captures volatility clustering with a smoother path compared to GARCH, reflecting its stochastic rather than deterministic volatility dynamics.

| Parameter | Posterior Mean | Posterior Std |
|-----------|---------------|---------------|
| $\alpha_0$  | ~0.50         | ~0.15         |
| $\alpha_1$  | ~0.80         | ~0.04         |
| $\sigma_v$  | ~0.10         | ~0.02         |

*(Exact values depend on the MCMC run; see the notebook for reproducible estimates.)*

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd sv-volatility-model

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

**Run the full pipeline from the notebook:**

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

**Or use the modules directly in Python:**

```python
from src.data_loader import load_returns
from src.sv_model import estimate_sv
from src.garch_model import fit_garch
from src.var_backtest import backtest_var, print_backtest_summary
from src.config import SVConfig

# Load data
df, returns = load_returns()

# Estimate SV model
sv_result = estimate_sv(returns, config=SVConfig(n_gibbs=5000))

# GARCH baseline
_, garch_vol, _ = fit_garch(returns)

# VaR backtest
sv_vol = sv_result.posterior_mean_vol()
var_result = backtest_var(returns, sv_vol, alpha=0.05, model_name="SV")
print_backtest_summary(var_result)
```

## Repository Structure

```
sv-volatility-model/
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters and configuration dataclasses
│   ├── data_loader.py     # Data loading and log return computation
│   ├── garch_model.py     # GARCH(1,1) estimation via arch package
│   ├── sv_model.py        # SV model MCMC estimation (Gibbs + Griddy-Gibbs)
│   ├── var_backtest.py    # VaR computation, Kupiec & Christoffersen tests
│   └── plotting.py        # Visualization functions
├── notebooks/
│   └── analysis.ipynb     # Full estimation and evaluation pipeline
├── data/
│   └── BRK-B.csv          # Berkshire Hathaway B daily prices (2018–2023)
├── figures/                # Generated plots (created by the notebook)
├── thesis/
│   └── Kai_Koepchen_7385179.pdf   # Bachelor's thesis PDF
├── matlab_original/        # Original MATLAB implementation (reference)
│   ├── MCMC_estimation.m
│   ├── plot_results.m
│   └── VaR_Computation.m
├── README.md
├── requirements.txt
└── .gitignore
```

## References

- Koepchen, K. (2023). *Bayesian Analysis of Stochastic Volatility Models for Stock Returns*. Bachelor's thesis, University of Cologne.
- Ritter, C. & Tanner, M.A. (1992). Facilitating the Gibbs Sampler: The Gibbs Stopper and the Griddy-Gibbs Sampler. *Journal of the American Statistical Association*.
- Tsay, R.S. (2010). *Analysis of Financial Time Series*. 3rd ed., Wiley.
- Kupiec, P.H. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*.
- Christoffersen, P.F. (1998). Evaluating Interval Forecasts. *International Economic Review*.

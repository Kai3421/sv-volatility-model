"""Configuration and default hyperparameters for the SV model estimation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SVConfig:
    """Configuration for the Stochastic Volatility MCMC estimation.

    Attributes:
        n_gibbs: Number of Gibbs sampling iterations.
        burnin_frac: Fraction of iterations to discard as burn-in.
        alpha0_start: Starting value for alpha_0 (intercept of log-volatility AR(1)).
        alpha1_start: Starting value for alpha_1 (persistence of log-volatility AR(1)).
        sigma_start: Starting value for sigma_v (volatility of log-volatility innovations).
        m: Degrees of freedom hyperparameter for the inverse-chi-squared prior on sigma_v^2.
        alpha_prior_mean: Prior mean for [alpha_0, alpha_1].
        alpha_prior_cov: Prior covariance matrix for [alpha_0, alpha_1].
        n_grid: Number of grid points for the Griddy-Gibbs sampler.
        grid_lower: Lower bound of the Griddy-Gibbs grid (h_t space).
        grid_upper: Upper bound of the Griddy-Gibbs grid (h_t space).
        seed: Random seed for reproducibility.
    """

    n_gibbs: int = 5000
    burnin_frac: float = 0.5
    alpha0_start: float = 0.5
    alpha1_start: float = 0.8
    sigma_start: float = 0.1
    m: int = 10
    alpha_prior_mean: list = field(default_factory=lambda: [0.5, 0.8])
    alpha_prior_cov: list = field(default_factory=lambda: [[0.25, 0.0], [0.0, 0.04]])
    n_grid: int = 3000
    grid_lower: float = 0.001
    grid_upper: float = 10.0
    seed: int = 123


@dataclass
class DataConfig:
    """Configuration for data loading.

    Attributes:
        data_path: Path to the CSV file with stock price data.
        ticker: Ticker symbol for labeling.
        open_col: Column name for opening prices.
        close_col: Column name for closing prices.
    """

    data_path: Optional[str] = None
    ticker: str = "BRK-B"
    open_col: str = "Open"
    close_col: str = "Close"


@dataclass
class VaRConfig:
    """Configuration for Value-at-Risk computation and backtesting.

    Attributes:
        alpha: VaR confidence level (e.g. 0.05 for 5% VaR).
    """

    alpha: float = 0.05


def get_default_data_path() -> Path:
    """Return path to the sample BRK-B data shipped with the repo."""
    return Path(__file__).resolve().parent.parent / "data" / "BRK-B.csv"

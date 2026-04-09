"""Plotting utilities for SV model diagnostics and VaR visualization."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from .sv_model import SVResult
from .var_backtest import VaRResult


def plot_returns(
    returns: np.ndarray,
    ticker: str = "BRK-B",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot the log return series.

    Args:
        returns: 1-D array of log returns.
        ticker: Ticker symbol for the title.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(returns, linewidth=0.7)
    ax.set_xlabel("Trading days")
    ax.set_ylabel(r"$r_t$")
    ax.set_title(f"Log returns of {ticker}")
    ax.set_xlim(0, len(returns))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_acf(
    returns: np.ndarray,
    nlags: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot sample ACF of r_t, r_t^2, and |r_t|.

    Args:
        returns: 1-D array of log returns.
        nlags: Number of lags to display.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("Sample ACF of log returns and transformations", fontsize=14)

    sm_plot_acf(returns, lags=nlags, ax=axes[0], title="")
    axes[0].set_ylabel(r"ACF of $r_t$")

    sm_plot_acf(returns**2, lags=nlags, ax=axes[1], title="")
    axes[1].set_ylabel(r"ACF of $r_t^2$")

    sm_plot_acf(np.abs(returns), lags=nlags, ax=axes[2], title="")
    axes[2].set_ylabel(r"ACF of $|r_t|$")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_traceplots(
    result: SVResult,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot MCMC trace plots for alpha_0, alpha_1, and sigma_v.

    Args:
        result: SVResult from the MCMC estimation.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("MCMC Trace Plots", fontsize=14)

    params = [
        (result.alpha0_draws, r"$\alpha_0$"),
        (result.alpha1_draws, r"$\alpha_1$"),
        (result.sigma_v_draws, r"$\sigma_v$"),
    ]
    for ax, (draws, label) in zip(axes, params):
        ax.plot(draws, linewidth=0.5)
        ax.set_ylabel(label)
        ax.set_xlim(0, len(draws))
    axes[-1].set_xlabel("Iteration")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_inferred_volatility(
    returns: np.ndarray,
    volatility: np.ndarray,
    model_name: str = "SV",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot returns overlaid with inferred volatilities.

    Args:
        returns: 1-D array of log returns.
        volatility: 1-D array of inferred volatilities.
        model_name: Label for the volatility model.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(returns, linewidth=0.5, alpha=0.7, label="Log returns")
    ax.plot(volatility, linewidth=1.0, label=f"Inferred volatility ({model_name})")
    ax.set_xlabel("Trading days")
    ax.set_ylabel(r"$r_t$, $\sqrt{h_t}$")
    ax.set_title(f"Log returns and inferred volatilities ({model_name})")
    ax.legend(loc="upper left")
    ax.set_xlim(0, len(returns))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_var_backtest(
    returns: np.ndarray,
    var_result: VaRResult,
    volatility: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot returns, inferred volatility, and VaR estimates with violations highlighted.

    Args:
        returns: 1-D array of log returns.
        var_result: VaRResult from the backtesting.
        volatility: 1-D array of conditional volatilities.
        save_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(returns, linewidth=0.5, alpha=0.6, label="Log returns", color="steelblue")
    ax.plot(volatility, linewidth=0.8, label=f"Volatility ({var_result.model_name})", color="orange")
    ax.plot(var_result.var_estimates, linewidth=0.8, label=f"VaR 5% ({var_result.model_name})", color="red", linestyle="--")

    # Highlight violations
    violations = np.where(var_result.hit_sequence == 1)[0]
    ax.scatter(violations, returns[violations], color="red", s=10, zorder=5, label="VaR violations")

    ax.set_xlabel("Trading days")
    ax.set_ylabel(r"$r_t$")
    ax.set_title(f"VaR Backtest: {var_result.model_name}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, len(returns))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

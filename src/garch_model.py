"""GARCH(1,1) baseline model estimation using maximum likelihood."""

import numpy as np
from arch import arch_model
from typing import Tuple


def fit_garch(returns: np.ndarray) -> Tuple[object, np.ndarray, np.ndarray]:
    """Fit a GARCH(1,1) model to the return series via maximum likelihood.

    The model specification is:
        r_t = mu + e_t,  e_t ~ N(0, sigma_t^2)
        sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2

    Args:
        returns: 1-D array of log returns (in percentage points).

    Returns:
        Tuple of (fitted model result, conditional volatility, conditional variance).
    """
    model = arch_model(returns, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    result = model.fit(disp="off")
    cond_var = result.conditional_volatility ** 2
    cond_vol = result.conditional_volatility
    return result, cond_vol, cond_var


def garch_residual_diagnostics(returns: np.ndarray, cond_vol: np.ndarray) -> dict:
    """Compute standardized residuals and Ljung-Box test on squared residuals.

    Args:
        returns: 1-D array of log returns.
        cond_vol: 1-D array of conditional volatilities from GARCH.

    Returns:
        Dictionary with keys 'standardized_residuals', 'lb_stat', 'lb_pvalue'.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    std_resid = returns / cond_vol
    sq_resid = std_resid ** 2
    lb = acorr_ljungbox(sq_resid, lags=[10], return_df=True)
    return {
        "standardized_residuals": std_resid,
        "lb_stat": lb["lb_stat"].values[0],
        "lb_pvalue": lb["lb_pvalue"].values[0],
    }

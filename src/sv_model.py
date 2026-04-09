"""Stochastic Volatility model estimation via Gibbs sampling with Griddy-Gibbs data augmentation.

The model:
    r_t = sqrt(h_t) * e_t,          e_t ~ N(0, 1)
    ln(h_t) = alpha_0 + alpha_1 * ln(h_{t-1}) + v_t,   v_t ~ N(0, sigma_v^2)

The latent log-volatilities ln(h_t) are sampled using the Griddy-Gibbs sampler
(Ritter & Tanner, 1992), while the parameters (alpha_0, alpha_1) and sigma_v^2
are drawn from their conjugate conditional posteriors.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .config import SVConfig


@dataclass
class SVResult:
    """Container for MCMC estimation results.

    Attributes:
        alpha0_draws: Array of shape (n_gibbs+1,) with alpha_0 draws.
        alpha1_draws: Array of shape (n_gibbs+1,) with alpha_1 draws.
        sigma_v_draws: Array of shape (n_gibbs+1,) with sigma_v draws.
        h_draws: Array of shape (T, n_gibbs+1) with volatility draws h_t.
        config: The SVConfig used for estimation.
    """

    alpha0_draws: np.ndarray
    alpha1_draws: np.ndarray
    sigma_v_draws: np.ndarray
    h_draws: np.ndarray
    config: SVConfig

    @property
    def burnin(self) -> int:
        """Number of burn-in iterations."""
        return int(self.config.n_gibbs * self.config.burnin_frac)

    def posterior_mean_log_vol(self) -> np.ndarray:
        """Compute posterior mean of ln(h_t) after discarding burn-in.

        Returns:
            1-D array of length T with posterior mean log-volatilities.
        """
        burnin = self.burnin
        log_h = np.log(self.h_draws[:, burnin:])
        return log_h.mean(axis=1)

    def posterior_mean_vol(self) -> np.ndarray:
        """Compute posterior mean of h_t (volatility level) after burn-in.

        Returns:
            1-D array of length T with posterior mean volatilities.
        """
        return np.exp(self.posterior_mean_log_vol())

    def parameter_summary(self) -> dict:
        """Compute posterior mean and std for all parameters after burn-in.

        Returns:
            Dictionary with keys 'alpha0', 'alpha1', 'sigma_v', each mapping
            to a dict with 'mean' and 'std'.
        """
        b = self.burnin
        summary = {}
        for name, draws in [
            ("alpha0", self.alpha0_draws),
            ("alpha1", self.alpha1_draws),
            ("sigma_v", self.sigma_v_draws),
        ]:
            post = draws[b:]
            summary[name] = {"mean": float(np.mean(post)), "std": float(np.std(post))}
        return summary


def _griddy_gibbs_sample_ht(
    t: int,
    T: int,
    r: np.ndarray,
    h_t: np.ndarray,
    alpha0: float,
    alpha1: float,
    sigma_v: float,
    grid: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Draw h_t from its conditional posterior using the Griddy-Gibbs sampler.

    For 1 < t < T, the conditional posterior (up to a normalizing constant) is:
        p(h_t | ...) ~ h_t^{-3/2} * exp(-r_t^2 / (2*h_t) - (ln(h_t) - mu_t)^2 / (2*sig_t^2))
    where mu_t and sig_t^2 depend on neighboring log-volatilities.

    For t = T, the forward-looking term drops out.

    Args:
        t: Time index (0-based).
        T: Total number of observations.
        r: Return series.
        h_t: Current volatility path.
        alpha0: Current alpha_0 draw.
        alpha1: Current alpha_1 draw.
        sigma_v: Current sigma_v draw.
        grid: Grid of h_t values for numerical integration.
        rng: Numpy random generator.

    Returns:
        Sampled value of h_t.
    """
    r_t = r[t]

    if t < T - 1:
        # Interior point: conditioned on h_{t-1} and h_{t+1}
        mu_t = (alpha0 * (1 - alpha1) + alpha1 * (np.log(h_t[t + 1]) + np.log(h_t[t - 1]))) / (
            1 + alpha1**2
        )
        sig_sq_t = sigma_v**2 / (1 + alpha1**2)
    else:
        # Terminal point t = T: no h_{t+1}
        mu_t = alpha0 + alpha1 * np.log(h_t[t - 1])
        sig_sq_t = sigma_v**2

    # Evaluate unnormalized conditional posterior on grid
    log_grid = np.log(grid)
    log_p = -1.5 * log_grid - r_t**2 / (2 * grid) - (log_grid - mu_t) ** 2 / (2 * sig_sq_t)
    # Shift for numerical stability
    log_p -= log_p.max()
    p = np.exp(log_p)
    p /= p.sum()

    # Draw from multinomial (equivalent to inverse CDF sampling on the grid)
    idx = rng.choice(len(grid), p=p)
    return grid[idx]


def _sample_sigma_v(
    h_t: np.ndarray,
    alpha0: float,
    alpha1: float,
    sigma_v_prev: float,
    m: int,
    T: int,
    rng: np.random.Generator,
) -> float:
    """Draw sigma_v from its conditional posterior (inverse-chi-squared).

    The conditional posterior for sigma_v follows from the conjugate
    inverse-chi-squared prior combined with the AR(1) log-volatility likelihood.

    Args:
        h_t: Current volatility path of length T.
        alpha0: Current alpha_0 draw.
        alpha1: Current alpha_1 draw.
        sigma_v_prev: Previous sigma_v draw (used in the prior scaling).
        m: Prior degrees of freedom hyperparameter.
        T: Number of observations.
        rng: Numpy random generator.

    Returns:
        New draw for sigma_v.
    """
    log_h = np.log(h_t)
    vt = log_h[2:] - alpha0 - alpha1 * log_h[1:-1]
    sum_vt_sq = np.sum(vt**2)
    scale = m * sigma_v_prev + sum_vt_sq
    draw = rng.chisquare(df=m + T - 1)
    return draw / scale


def _sample_alpha(
    h_t: np.ndarray,
    sigma_v: float,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    T: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Draw (alpha_0, alpha_1) from their conjugate normal conditional posterior.

    Given the AR(1) structure of log-volatilities and a normal prior on alpha,
    the conditional posterior is also normal.

    Args:
        h_t: Current volatility path of length T.
        sigma_v: Current sigma_v draw.
        prior_mean: Prior mean vector [alpha_0, alpha_1].
        prior_cov: Prior covariance matrix (2x2).
        T: Number of observations.
        rng: Numpy random generator.

    Returns:
        Tuple (alpha_0, alpha_1).
    """
    log_h = np.log(h_t)
    # Design matrix: z_t = [1, ln(h_{t-1})] for t = 2, ..., T-1 (0-indexed: t=2..T-1)
    Z = np.column_stack([np.ones(T - 2), log_h[1:-1]])
    y = log_h[2:]

    prior_cov_inv = np.linalg.inv(prior_cov)
    ZtZ = Z.T @ Z
    Zty = Z.T @ y

    post_cov = np.linalg.inv(ZtZ / sigma_v + prior_cov_inv)
    post_mean = post_cov @ (Zty / sigma_v + prior_cov_inv @ prior_mean)

    alpha_draw = rng.multivariate_normal(post_mean, post_cov)
    return float(alpha_draw[0]), float(alpha_draw[1])


def estimate_sv(returns: np.ndarray, config: SVConfig | None = None) -> SVResult:
    """Estimate the Stochastic Volatility model via MCMC (Gibbs + Griddy-Gibbs).

    The sampler alternates between:
      1. Drawing latent volatilities h_t via Griddy-Gibbs (data augmentation).
      2. Drawing sigma_v from its inverse-chi-squared conditional posterior.
      3. Drawing (alpha_0, alpha_1) from their normal conditional posterior.

    Initial volatilities are obtained from a GARCH(1,1) fit.

    Args:
        returns: 1-D array of log returns (in percentage points).
        config: Model configuration. If None, uses defaults.

    Returns:
        SVResult containing all MCMC draws and configuration.
    """
    if config is None:
        config = SVConfig()

    rng = np.random.default_rng(config.seed)
    T = len(returns)

    # Initialize h_t from GARCH(1,1) conditional volatilities
    from .garch_model import fit_garch

    _, garch_vol, _ = fit_garch(returns)
    h_t = garch_vol.copy()

    # Set up storage
    n = config.n_gibbs
    alpha0_draws = np.zeros(n + 1)
    alpha1_draws = np.zeros(n + 1)
    sigma_v_draws = np.zeros(n + 1)
    h_draws = np.zeros((T, n + 1))

    alpha0_draws[0] = config.alpha0_start
    alpha1_draws[0] = config.alpha1_start
    sigma_v_draws[0] = config.sigma_start
    h_draws[:, 0] = h_t

    # Grid for Griddy-Gibbs
    grid = np.linspace(config.grid_lower, config.grid_upper, config.n_grid)

    prior_mean = np.array(config.alpha_prior_mean)
    prior_cov = np.array(config.alpha_prior_cov)

    # Gibbs sampler
    for ii in range(n):
        a0 = alpha0_draws[ii]
        a1 = alpha1_draws[ii]
        sv = sigma_v_draws[ii]

        # Step 1: Sample latent volatilities via Griddy-Gibbs
        for t in range(1, T):
            h_t[t] = _griddy_gibbs_sample_ht(t, T, returns, h_t, a0, a1, sv, grid, rng)

        h_draws[:, ii + 1] = h_t

        # Step 2: Sample sigma_v
        sigma_v_draws[ii + 1] = _sample_sigma_v(h_t, a0, a1, sv, config.m, T, rng)

        # Step 3: Sample (alpha_0, alpha_1)
        a0_new, a1_new = _sample_alpha(h_t, sigma_v_draws[ii + 1], prior_mean, prior_cov, T, rng)
        alpha0_draws[ii + 1] = a0_new
        alpha1_draws[ii + 1] = a1_new

        if (ii + 1) % 500 == 0:
            print(f"  MCMC iteration {ii + 1}/{n}")

    return SVResult(
        alpha0_draws=alpha0_draws,
        alpha1_draws=alpha1_draws,
        sigma_v_draws=sigma_v_draws,
        h_draws=h_draws,
        config=config,
    )

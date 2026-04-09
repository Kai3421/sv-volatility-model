"""Value-at-Risk computation and backtesting via Kupiec and Christoffersen tests.

This module implements the VaR evaluation framework from Chapter 5 of the thesis.
Given volatility estimates from the SV model or GARCH(1,1), it computes VaR
quantiles and applies formal coverage tests.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VaRResult:
    """Container for VaR estimates and backtest results.

    Attributes:
        var_estimates: 1-D array of VaR estimates for each trading day.
        hit_sequence: Binary array (1 if return < VaR, 0 otherwise).
        kupiec: Dict with Kupiec unconditional coverage test results.
        christoffersen: Dict with Christoffersen independence test results.
        cc_test: Dict with conditional coverage (combined) test results.
        model_name: Label for the model (e.g. "SV" or "GARCH(1,1)").
    """

    var_estimates: np.ndarray
    hit_sequence: np.ndarray
    kupiec: dict
    christoffersen: dict
    cc_test: dict
    model_name: str


def compute_var(volatility: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Compute Value-at-Risk estimates assuming conditional normality.

    Under the model r_t | F_{t-1} ~ N(0, h_t), the alpha-level VaR is:
        VaR_t(alpha) = Phi^{-1}(alpha) * sqrt(h_t)

    Args:
        volatility: 1-D array of conditional volatilities (sqrt(h_t)).
        alpha: VaR level (left-tail probability). Default 0.05.

    Returns:
        1-D array of VaR estimates (negative values indicating losses).
    """
    return stats.norm.ppf(alpha) * volatility


def _hit_sequence(returns: np.ndarray, var_estimates: np.ndarray) -> np.ndarray:
    """Compute the VaR hit (violation) sequence.

    Args:
        returns: Observed log returns.
        var_estimates: VaR estimates.

    Returns:
        Binary array: 1 where return falls below VaR, 0 otherwise.
    """
    return (returns <= var_estimates).astype(int)


def kupiec_test(hits: np.ndarray, alpha: float) -> dict:
    """Kupiec (1995) unconditional coverage test (proportion of failures).

    Tests H0: E[I_t] = alpha against H1: E[I_t] != alpha using the
    likelihood ratio statistic:
        LR_uc = -2 * ln(L0 / L1)
    which is asymptotically chi-squared(1) under H0.

    Args:
        hits: Binary hit sequence.
        alpha: Nominal VaR level.

    Returns:
        Dict with 'n_violations', 'violation_rate', 'lr_statistic', 'critical_value',
        'p_value', and 'reject' (bool at 5% significance).
    """
    T = len(hits)
    n1 = int(hits.sum())
    n0 = T - n1
    pi_hat = n1 / T

    # Avoid log(0)
    if n1 == 0 or n1 == T:
        lr = 0.0
    else:
        log_L0 = n0 * np.log(1 - alpha) + n1 * np.log(alpha)
        log_L1 = n0 * np.log(1 - pi_hat) + n1 * np.log(pi_hat)
        lr = -2 * (log_L0 - log_L1)

    crit = stats.chi2.ppf(0.95, df=1)
    p_value = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "n_violations": n1,
        "violation_rate": pi_hat,
        "lr_statistic": float(lr),
        "critical_value": float(crit),
        "p_value": float(p_value),
        "reject": bool(lr > crit),
    }


def christoffersen_test(hits: np.ndarray) -> dict:
    """Christoffersen (1998) independence test on the hit sequence.

    Tests whether VaR violations are serially independent by comparing a
    first-order Markov model against an i.i.d. Bernoulli model.

    The likelihood ratio statistic:
        LR_ind = -2 * ln(L_iid / L_markov)
    is asymptotically chi-squared(1) under H0 (independence).

    Args:
        hits: Binary hit sequence.

    Returns:
        Dict with transition counts, 'lr_statistic', 'critical_value',
        'p_value', and 'reject'.
    """
    transitions = hits[1:] - hits[:-1]

    n01 = int(np.sum(transitions == 1))
    n10 = int(np.sum(transitions == -1))
    # When transition is 0, check whether we stayed in state 0 or state 1
    no_change_mask = transitions == 0
    n11 = int(np.sum(hits[1:][no_change_mask] == 1))
    n00 = int(np.sum(hits[1:][no_change_mask] == 0))

    # Markov transition probabilities
    denom_0 = n00 + n01
    denom_1 = n10 + n11

    if denom_0 == 0 or denom_1 == 0 or n00 == 0 or n01 == 0 or n10 == 0 or n11 == 0:
        # Degenerate case: not enough transitions to test
        return {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "lr_statistic": 0.0,
            "critical_value": float(stats.chi2.ppf(0.95, df=1)),
            "p_value": 1.0,
            "reject": False,
        }

    # Log-likelihood under Markov model
    log_L1 = (
        n00 * np.log(n00 / denom_0)
        + n01 * np.log(n01 / denom_0)
        + n10 * np.log(n10 / denom_1)
        + n11 * np.log(n11 / denom_1)
    )

    # Log-likelihood under i.i.d. model
    pi_hat = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_L0 = (n00 + n10) * np.log(1 - pi_hat) + (n01 + n11) * np.log(pi_hat)

    lr = -2 * (log_L0 - log_L1)
    crit = stats.chi2.ppf(0.95, df=1)
    p_value = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "lr_statistic": float(lr),
        "critical_value": float(crit),
        "p_value": float(p_value),
        "reject": bool(lr > crit),
    }


def conditional_coverage_test(hits: np.ndarray, alpha: float) -> dict:
    """Christoffersen conditional coverage test (Kupiec + independence combined).

    LR_cc = LR_uc + LR_ind ~ chi-squared(2) under H0.

    Args:
        hits: Binary hit sequence.
        alpha: Nominal VaR level.

    Returns:
        Dict with 'lr_statistic', 'critical_value', 'p_value', 'reject'.
    """
    kup = kupiec_test(hits, alpha)
    ind = christoffersen_test(hits)
    lr_cc = kup["lr_statistic"] + ind["lr_statistic"]
    crit = stats.chi2.ppf(0.95, df=2)
    p_value = 1 - stats.chi2.cdf(lr_cc, df=2)

    return {
        "lr_statistic": float(lr_cc),
        "critical_value": float(crit),
        "p_value": float(p_value),
        "reject": bool(lr_cc > crit),
    }


def backtest_var(
    returns: np.ndarray,
    volatility: np.ndarray,
    alpha: float = 0.05,
    model_name: str = "Model",
) -> VaRResult:
    """Run the full VaR backtesting pipeline.

    Computes VaR estimates from volatility, constructs the hit sequence,
    and runs Kupiec, Christoffersen independence, and conditional coverage tests.

    Args:
        returns: 1-D array of observed log returns.
        volatility: 1-D array of conditional volatilities.
        alpha: VaR level. Default 0.05.
        model_name: Label for the model.

    Returns:
        VaRResult with all test outcomes.
    """
    var_est = compute_var(volatility, alpha)
    hits = _hit_sequence(returns, var_est)
    kup = kupiec_test(hits, alpha)
    chris = christoffersen_test(hits)
    cc = conditional_coverage_test(hits, alpha)

    return VaRResult(
        var_estimates=var_est,
        hit_sequence=hits,
        kupiec=kup,
        christoffersen=chris,
        cc_test=cc,
        model_name=model_name,
    )


def print_backtest_summary(result: VaRResult) -> None:
    """Print a formatted summary of VaR backtest results.

    Args:
        result: VaRResult from backtest_var.
    """
    print(f"\n{'='*60}")
    print(f"VaR Backtest Results: {result.model_name}")
    print(f"{'='*60}")
    print(f"  Violations: {result.kupiec['n_violations']} / {len(result.hit_sequence)}")
    print(f"  Violation rate: {result.kupiec['violation_rate']:.4f}")
    print()
    print(f"  Kupiec UC test:        LR = {result.kupiec['lr_statistic']:.4f}  "
          f"(crit = {result.kupiec['critical_value']:.4f}, p = {result.kupiec['p_value']:.4f})  "
          f"{'REJECT' if result.kupiec['reject'] else 'PASS'}")
    print(f"  Christoffersen ind:    LR = {result.christoffersen['lr_statistic']:.4f}  "
          f"(crit = {result.christoffersen['critical_value']:.4f}, p = {result.christoffersen['p_value']:.4f})  "
          f"{'REJECT' if result.christoffersen['reject'] else 'PASS'}")
    print(f"  Conditional coverage:  LR = {result.cc_test['lr_statistic']:.4f}  "
          f"(crit = {result.cc_test['critical_value']:.4f}, p = {result.cc_test['p_value']:.4f})  "
          f"{'REJECT' if result.cc_test['reject'] else 'PASS'}")
    print(f"{'='*60}\n")

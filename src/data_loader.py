"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from .config import DataConfig, get_default_data_path


def load_returns(config: DataConfig | None = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load stock price data and compute log returns.

    Log returns are computed as r_t = ln(Close_t / Open_t) * 100,
    following the convention used in the thesis.

    Args:
        config: Data configuration. If None, uses defaults with the sample dataset.

    Returns:
        Tuple of (raw DataFrame, log returns as 1-D numpy array).
    """
    if config is None:
        config = DataConfig()

    data_path = Path(config.data_path) if config.data_path else get_default_data_path()
    df = pd.read_csv(data_path, parse_dates=["Date"])
    log_returns = np.log(df[config.close_col].values / df[config.open_col].values) * 100
    return df, log_returns

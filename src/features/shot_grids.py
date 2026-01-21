# src/features/shot_grids.py
import numpy as np
import pandas as pd

FT_TO_IN = 12

DEFAULT_BINS = [0, 2, 4, 6, np.inf]
DEFAULT_LABELS = ["Super Contested", "Contested", "Slightly Contested", "Wide Open"]

def contested_fg_grids(
    shots: pd.DataFrame,
    n_bins: int = 10,
    x_min: float = -25 * FT_TO_IN,
    x_max: float =  25 * FT_TO_IN,
    y_min: float = 0 * FT_TO_IN,
    y_max: float = 47 * FT_TO_IN,
    dist_col: str = "CLOSE_DEF_DIST",
    x_col: str = "LOC_X",
    y_col: str = "LOC_Y",
    made_col: str = "SHOT_MADE_FLAG",
    bins=DEFAULT_BINS,
    labels=DEFAULT_LABELS,
) -> dict[str, np.ndarray]:
    df = shots.copy()

    # 1) contested bucket
    df["contested_level"] = pd.cut(df[dist_col], bins=bins, labels=labels, right=True)

    # 2) grid edges + bin indices
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)

    df["x_bin"] = np.digitize(df[x_col], x_edges) - 1
    df["y_bin"] = np.digitize(df[y_col], y_edges) - 1
    df["x_bin"] = df["x_bin"].clip(0, n_bins - 1)
    df["y_bin"] = df["y_bin"].clip(0, n_bins - 1)

    # 3) aggregate FG% per contested level
    grids: dict[str, np.ndarray] = {}
    for level in labels:
        fg = np.full((n_bins, n_bins), np.nan, dtype=float)
        lvl = df[df["contested_level"] == level]
        for (y, x), g in lvl.groupby(["y_bin", "x_bin"]):
            fg[int(y), int(x)] = g[made_col].mean()
        grids[level] = fg

    return grids

import numpy as np

def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def openness(
    dmin: float,
    closing_speed: float,
    *,
    d0: float = 4.0,
    k_dist: float = 1.2,
    k_close: float = 0.6,
    closing_convention: str = "closing_positive",
) -> float:
    """
    Openness in (0,1).

    Parameters
    ----------
    dmin : float
        Defender distance (feet). Larger => more open.
    closing_speed : float
        Defender closing speed.
        - If closing_convention="closing_positive": positive means distance is DECREASING (defender closing in).
        - If closing_convention="deriv": this is d(distance)/dt, so negative means closing in.
    d0 : float
        Distance midpoint (feet) where openness ~0.5 if closing_speed is 0.
    k_dist : float
        Sensitivity to distance.
    k_close : float
        Sensitivity to closing speed.

    Returns
    -------
    float
        Openness score in (0,1).
    """
    if closing_convention == "deriv":
        # d(dist)/dt: negative means closing; convert to "closing_positive"
        closing_in = -closing_speed
    elif closing_convention == "closing_positive":
        closing_in = closing_speed
    else:
        raise ValueError("closing_convention must be 'closing_positive' or 'deriv'")

    z = k_dist * (dmin - d0) - k_close * closing_in
    return sigmoid(z)


def sample_grid_nearest(grid, xedges, yedges, x, y) -> float:
    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1
    ix = np.clip(ix, 0, grid.shape[0] - 1)
    iy = np.clip(iy, 0, grid.shape[1] - 1)
    return float(grid[ix, iy])


def shootability(speed: float, accel: float, v0: float = 10.0, a0: float = 20.0) -> float:
    return float(np.exp(-(speed / v0) ** 2 - (accel / a0) ** 2))


def ball_factor(dist_to_ball: float, r0: float = 6.0) -> float:
    return float(np.exp(-(dist_to_ball / r0) ** 2))


def compute_ist_from_maps(
    *,
    pid: int,
    x: float,
    y: float,
    maps_npz: dict,
    pid2row: dict,
    dmin: float,
    closing_speed: float,
    speed: float,
    accel: float,
    dist_to_ball: float = 0.0,
    use: str = "quality",
    closing_convention: str = "closing_positive",
    include_ball: bool = False,
) -> dict:
    """
    Compute IST and its components using NPZ-style maps.

    Returns a dict with IST, Q, O, S, B.
    """
    if int(pid) not in pid2row:
        return {"IST": np.nan, "Q": np.nan, "O": np.nan, "S": np.nan, "B": np.nan, "reason": "pid_not_in_maps"}

    i = pid2row[int(pid)]
    xedges = maps_npz["xedges"]
    yedges = maps_npz["yedges"]
    grid = maps_npz[use][i]

    Q = sample_grid_nearest(grid, xedges, yedges, x, y)
    O = openness(dmin, closing_speed, closing_convention=closing_convention)
    S = shootability(speed, accel)

    if include_ball:
        B = ball_factor(dist_to_ball)
    else:
        B = 1.0

    IST = float(Q * O * S * B)
    return {"IST": IST, "Q": Q, "O": O, "S": S, "B": float(B), "reason": "ok"}


def add_ist_column(df, maps, pid2row, use="quality"):
    out = df.copy()
    IST, Qs, Os, Ss = [], [], [], []
    for _, r in out.iterrows():
        res = compute_ist_from_maps(
            pid=int(r["PLAYER_ID"]),
            x=float(r["x_ft"]),      # feet coords
            y=float(r["y_ft"]),
            maps_npz=maps,
            pid2row=pid2row,
            dmin=float(r["close_def_dist_mean"]),
            closing_speed=float(r["close_def_closing_speed_mean"]),
            speed=float(r["shooter_speed_mean"]),
            accel=float(r["shooter_accel_mean"]),
            closing_convention="deriv",
            include_ball=False,
            use=use
        )
        IST.append(res["IST"]); Qs.append(res["Q"]); Os.append(res["O"]); Ss.append(res["S"])
    out["IST"] = IST
    out["IST_Q"] = Qs
    out["IST_O"] = Os
    out["IST_S"] = Ss
    return out
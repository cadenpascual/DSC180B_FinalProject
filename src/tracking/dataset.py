import numpy as np
import xarray as xr
from .tensorize import event_to_tensor_offense

# Dataset with offensive players + ball
def build_offensive_dataset(game, max_frames=150):
    plays = []

    for event in game:
        if not event.get("frames"):
            continue

        tensor = event_to_tensor_offense(event, max_frames=max_frames)
        if tensor.size > 0:
            plays.append(tensor)

    if not plays:
        return None

    P = len(plays)  # number of events
    T = max(p.shape[0] for p in plays)  # max time steps
    N = plays[0].shape[1]  # number of offensive players

    # Initialize array
    data = np.zeros((P, T, N, 2))

    # Fill array
    for i, p in enumerate(plays):
        data[i, :p.shape[0]] = p

    # Build xarray dataset
    dataset = xr.Dataset(
        data_vars={"positions": (["play", "time", "player", "coord"], data)},
        coords={
            "play": np.arange(P),
            "time": np.arange(T),
            "player": np.arange(N),
            "coord": ["x", "y"]
        }
    )

    return dataset
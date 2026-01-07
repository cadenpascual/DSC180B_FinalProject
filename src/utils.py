import numpy as np
import xarray as xr        

# JSON -> Tensor (3D array: Moment, Players, (x,y) Coords)
def event_to_tensor(event, max_frames=None):
    moments = event["moments"]
    if max_frames:
        moments = moments[:max_frames]

    T = len(moments)
    traj = np.zeros((T, 10, 2))

    for t, m in enumerate(moments):
        players = m[5][1:]  # skip ball
        for i, p in enumerate(players):
            traj[t, i, 0] = p[2]  # x
            traj[t, i, 1] = p[3]  # y

    return traj


def build_dataset(game, max_frames=150):
    plays = []
    for event in game["events"]:
        if "moments" not in event or len(event["moments"]) < 10:
            continue
        plays.append(event_to_tensor(event, max_frames))

    P = len(plays)
    T = max(p.shape[0] for p in plays)

    data = np.zeros((P, T, 10, 2))

    for i, p in enumerate(plays):
        data[i, :p.shape[0]] = p

    return xr.Dataset(
        data_vars={
            "positions": (["play", "time", "player", "coord"], data)
        },
        coords={
            "play": np.arange(P),
            "time": np.arange(T),
            "player": np.arange(10),
            "coord": ["x", "y"]
        }
    )




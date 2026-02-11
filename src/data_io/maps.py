# ============================
# src/io/maps.py
# ============================
from pathlib import Path
import numpy as np


def save_maps_npz(path, maps):
    """
    Fastest storage: one compressed NPZ.
    Assumes all players share xedges/yedges (true if you built with one grid).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    any_pid = next(iter(maps))
    xedges = np.asarray(maps[any_pid]["xedges"], dtype=np.float32)
    yedges = np.asarray(maps[any_pid]["yedges"], dtype=np.float32)

    player_ids = np.asarray(list(maps.keys()), dtype=np.int64)

    density = np.stack([np.asarray(maps[int(pid)]["density"]) for pid in player_ids]).astype(np.float32)
    quality = np.stack([np.asarray(maps[int(pid)]["quality"]) for pid in player_ids]).astype(np.float32)
    impact  = np.stack([np.asarray(maps[int(pid)]["impact"])  for pid in player_ids]).astype(np.float32)

    attempt_count = np.asarray([maps[int(pid)]["attempt_count"] for pid in player_ids], dtype=np.int32)

    np.savez_compressed(
        path,
        xedges=xedges,
        yedges=yedges,
        player_ids=player_ids,
        attempt_count=attempt_count,
        density=density,
        quality=quality,
        impact=impact,
    )


def load_maps_npz(path):
    """
    Returns:
      maps_npz: dict of arrays
      pid2row: dict PLAYER_ID -> row index
    """
    z = np.load(path, allow_pickle=False)
    maps_npz = {
        "xedges": z["xedges"],
        "yedges": z["yedges"],
        "player_ids": z["player_ids"],
        "attempt_count": z["attempt_count"],
        "density": z["density"],
        "quality": z["quality"],
        "impact": z["impact"],
    }
    pid2row = {int(pid): i for i, pid in enumerate(maps_npz["player_ids"])}
    return maps_npz, pid2row
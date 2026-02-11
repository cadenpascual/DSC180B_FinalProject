# ============================
# src/utils/players.py
# ============================
import pandas as pd

def find_player_id(meta: pd.DataFrame, name: str) -> int:
    """
    Case-insensitive substring match to find PLAYER_ID by name.
    """
    row = meta[meta["PLAYER_NAME"].str.contains(name, case=False, regex=False)]
    if row.empty:
        raise ValueError(f"No player found matching '{name}'")
    return int(row["PLAYER_ID"].iloc[0])

def maps_npz_player_dict(maps_npz, pid2row, pid):
    i = pid2row[int(pid)]
    return {
        "xedges": maps_npz["xedges"],
        "yedges": maps_npz["yedges"],
        "density": maps_npz["density"][i],
        "quality": maps_npz["quality"][i],
        "impact":  maps_npz["impact"][i],
    }
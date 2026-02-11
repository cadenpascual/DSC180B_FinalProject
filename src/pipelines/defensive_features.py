from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

from src.processing.sportvu_to_events import raw_sportvu_to_tracking_events
from src.processing.tracking_cleaning import dedupe_tracking_events
from src.processing.indexing import build_tracking_time_index

from src.processing.indexing import find_event_for_shot_by_clock, build_tracking_time_index
from src.features.defense_features import compute_pre_shot_defense_features


def build_shot_defense_features(
    game: dict,
    shots: pd.DataFrame,
    *,
    span_pad: float = 2.0,
    max_center_diff: float = 10.0,
    match: str = "closest",
    max_time_diff: float = 1.5,
    fps: int = 25,
    window_seconds: float = 1.0,
    smooth_window: int = 5,
) -> Tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """
    End-to-end:
      raw SportVU + season shots -> per-shot defensive features aligned to tracking frames.

    Returns:
      shots_feat: shots_g with added columns for defense features + debug alignment info
      tracking_events: cleaned tracking events (with frames)
      shot_alignment_debug: per-shot debug table (event idx, diffs, reasons)
    """
    game_id = int(game[0]['gameid'])

    # --- shots for this game ---
    shots_g = shots.loc[shots["GAME_ID"].astype(int) == game_id].copy()
    shots_g = shots_g.reset_index(drop=True)

    # --- tracking events ---
    tracking_events = raw_sportvu_to_tracking_events(game)
    tracking_events = dedupe_tracking_events(tracking_events)
    tracking_time_index = build_tracking_time_index(tracking_events)

    # (optional) if you want to reuse your pbp alignment logic, you can.
    # For now, we align shots directly by game clock using your time index.

    feats_rows = []
    debug_rows = []

    for i, shot in shots_g.iterrows():
        shot_gc = float(shot.get("game_clock", np.nan))
        quarter = int(shot.get("PERIOD", np.nan))
        offense_team_id = int(shot["TEAM_ID"])
        shooter_id = int(shot["PLAYER_ID"])

        ev_idx, info = find_event_for_shot_by_clock(tracking_time_index, game_id, quarter, shot_gc)

        debug = {
            "shot_row": int(i),
            "GAME_ID": game_id,
            "PERIOD": quarter,
            "shot_gc": shot_gc,
            "event_list_idx": ev_idx,
            **(info or {}),
        }

        if ev_idx is None:
            feats_rows.append({"shot_row": int(i), "error": "no_event_match"})
            debug_rows.append(debug)
            continue

        frames = tracking_events[int(ev_idx)]["frames"]

        release_idx, rinfo = build_tracking_time_index(
            event_frames=frames,
            shot_game_clock=shot_gc,
            match=match,
            max_time_diff=max_time_diff,
        )
        debug.update({f"release_{k}": v for k, v in (rinfo or {}).items()})
        debug["release_idx"] = release_idx

        feats = compute_pre_shot_defense_features(
            event_frames=frames,
            release_frame_idx=release_idx,
            shooter_id=shooter_id,
            offense_team_id=offense_team_id,
            fps=fps,
            window_seconds=window_seconds,
            smooth_window=smooth_window,
        )
        feats_rows.append({"shot_row": int(i), **feats})
        debug_rows.append(debug)

    feats_df = pd.DataFrame(feats_rows)
    debug_df = pd.DataFrame(debug_rows)

    shots_feat = shots_g.merge(feats_df, left_index=True, right_on="shot_row", how="left").drop(columns=["shot_row"])
    return shots_feat, tracking_events, debug_df
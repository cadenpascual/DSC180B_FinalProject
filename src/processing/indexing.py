# src/processing/indexing.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

def build_tracking_time_index(tracking_events: list[dict]) -> pd.DataFrame:
    rows = []
    for k, ev in enumerate(tracking_events):
        frames = ev.get("frames", [])
        if not frames:
            continue

        gc_raw = [fr.get("game_clock", np.nan) for fr in frames]
        gc = pd.to_numeric(gc_raw, errors="coerce").to_numpy()
        valid = ~np.isnan(gc)
        if valid.sum() < 2:
            continue

        gc_v = gc[valid]

        gc_start = float(np.max(gc_v))
        gc_end   = float(np.min(gc_v))
        gc_center = 0.5 * (gc_start + gc_end)
        gc_span = gc_start - gc_end

        # monotonicity check (countdown should mostly decrease)
        d = np.diff(gc_v)
        mono_frac = float(np.mean(d <= 0)) if len(d) else np.nan

        rows.append({
            "gameid": int(ev.get("gameid")) if ev.get("gameid") is not None else None,
            "quarter": int(ev.get("quarter")) if ev.get("quarter") is not None else None,
            "event_list_idx": k,
            "gc_start": gc_start,
            "gc_end": gc_end,
            "gc_center": gc_center,
            "gc_span": float(gc_span),
            "n_frames_total": int(len(frames)),
            "n_frames_gc": int(valid.sum()),
            "gc_monotone_frac": mono_frac,
        })

    df = pd.DataFrame(rows)
    # Optional: drop clearly broken segments
    if not df.empty:
        df = df[(df["gc_span"] > 0) & (df["n_frames_gc"] >= 10)]
    return df.reset_index(drop=True)
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

def find_event_for_shot_by_clock(
    event_index: pd.DataFrame,
    gameid: int,
    quarter: int,
    shot_game_clock: float,
    span_pad: float = 1.0,
    max_center_diff: float = 2.0,
    max_fallback_diff: float = 4.0,   # NEW: prevents bad matches
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Returns (event_list_idx, debug_info) where event_list_idx is an index into tracking_events, or None.
    """
    df = event_index[
        (event_index["gameid"] == gameid) &
        (event_index["quarter"] == quarter)
    ].copy()

    if df.empty:
        return None, {"reason": "no_events_for_game_quarter"}

    # Ensure numeric
    df["gc_start"] = pd.to_numeric(df["gc_start"], errors="coerce")
    df["gc_end"]   = pd.to_numeric(df["gc_end"], errors="coerce")
    df = df.dropna(subset=["gc_start", "gc_end"])
    if df.empty:
        return None, {"reason": "no_valid_gc_spans"}

    # Event span is [gc_end, gc_start] because clock counts down
    in_span = df[
        (shot_game_clock <= (df["gc_start"] + span_pad)) &
        (shot_game_clock >= (df["gc_end"]   - span_pad))
    ].copy()

    # Always compute center fields for selection
    def add_center_cols(d):
        d["gc_center"] = (d["gc_start"] + d["gc_end"]) / 2.0
        d["center_diff"] = np.abs(d["gc_center"] - shot_game_clock)
        return d

    if in_span.empty:
        # fallback: choose closest by CENTER (more stable than boundary)
        df2 = add_center_cols(df)
        best = df2.loc[df2["center_diff"].idxmin()]

        if float(best["center_diff"]) > max_fallback_diff:
            return None, {
                "reason": "fallback_center_diff_too_large",
                "center_diff": float(best["center_diff"]),
                "max_fallback_diff": float(max_fallback_diff),
                "gc_start": float(best["gc_start"]),
                "gc_end": float(best["gc_end"]),
            }

        return int(best["event_list_idx"]), {
            "reason": "fallback_closest_center",
            "center_diff": float(best["center_diff"]),
            "gc_start": float(best["gc_start"]),
            "gc_end": float(best["gc_end"]),
        }

    in_span = add_center_cols(in_span)
    best = in_span.loc[in_span["center_diff"].idxmin()]

    if float(best["center_diff"]) > max_center_diff:
        return None, {
            "reason": "center_diff_too_large",
            "center_diff": float(best["center_diff"]),
            "gc_start": float(best["gc_start"]),
            "gc_end": float(best["gc_end"]),
        }

    return int(best["event_list_idx"]), {
        "reason": "ok",
        "center_diff": float(best["center_diff"]),
        "gc_start": float(best["gc_start"]),
        "gc_end": float(best["gc_end"]),
    }
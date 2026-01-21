# src/data_sources/nba_api_shots.py
import time
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail

# Function to fetch league shot data with retries
def fetch_league_shots(
    season: str,
    season_type: str = "Regular Season",
    context_measure: str = "FGA",
    retries: int = 3,
    delay: float = 0.5,
) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries):
        try:
            df = shotchartdetail.ShotChartDetail(
                team_id=0,
                player_id=0,
                season_nullable=season,
                season_type_all_star=season_type,
                context_measure_simple=context_measure,
            ).get_data_frames()[0]
            time.sleep(delay)  # be polite
            return df
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise ConnectionError(f"Failed to fetch shots after {retries} attempts: {last_err}")

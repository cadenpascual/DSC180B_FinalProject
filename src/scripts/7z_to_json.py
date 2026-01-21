import os
import pandas as pd
from tqdm import tqdm

from src.data_io.archives import extract_and_load_json
from src.processing.summaries import summarize_game  # better separation

DATA_DIR = "data/raw/7z"
MAX_GAMES = 10

archives = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".7z")])

rows = []
processed_games = 0

for fname in tqdm(archives, desc="Processing games"):
    if processed_games >= MAX_GAMES:
        break

    archive_path = os.path.join(DATA_DIR, fname)
    data = extract_and_load_json(archive_path, tmp_root="data/tmp/extracted_json")

    if data is None:
        continue

    rows.append(summarize_game(data))
    processed_games += 1

df = pd.DataFrame(rows)
print(f"Processed {processed_games} valid games.")

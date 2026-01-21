import pandas as pd

# Safe casting functions
def safe_int(x):
    return int(x) if pd.notna(x) else None

def safe_float(x):
    return float(x) if x is not None else None
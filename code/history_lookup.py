"""Load and look up user claim history."""

from pathlib import Path

import pandas as pd


def load_history(path: str) -> pd.DataFrame:
    """Load user history CSV and return a dataframe."""
    return pd.read_csv(Path(path), dtype=str).fillna("")


def get_history(user_id: str, df: pd.DataFrame) -> dict:
    """Return the history row for user_id as a dict, or empty dict if not found."""
    matches = df[df["user_id"] == str(user_id)]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()

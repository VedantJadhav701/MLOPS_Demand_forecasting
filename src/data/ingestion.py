import pandas as pd


def load_data(path: str, date_col: str) -> pd.DataFrame:
    """
    Load dataset, parse date column, and sort for time-series consistency.
    """

    df = pd.read_csv(path)

    # parse date column
    df[date_col] = pd.to_datetime(df[date_col])

    # sort for time-series integrity
    df = df.sort_values(["product_id", "store_id", date_col])

    # basic validation
    required_cols = [
        "product_id",
        "store_id",
        date_col,
        "target_demand"
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df
import pandas as pd
import numpy as np


def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day_of_month"] = df[date_col].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["year"] = df[date_col].dt.year
    return df


def create_lag_features(
    df: pd.DataFrame,
    group_cols: list,
    target_col: str,
    lags: list
) -> pd.DataFrame:
    """Create lag features grouped by store/product."""
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
        if "price" in df.columns:
            df[f"price_lag_{lag}"] = df.groupby(group_cols)["price"].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame,
    group_cols: list,
    target_col: str,
    windows: list
) -> pd.DataFrame:
    """Create rolling statistics grouped by store/product using transform."""
    for window in windows:
        shifted = df.groupby(group_cols)[target_col].shift(1)

        # Rolling mean (per group via transform)
        df[f"rolling_mean_{window}"] = (
            shifted
            .groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        # Rolling std
        df[f"rolling_std_{window}"] = (
            shifted
            .groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
        # Rolling min
        df[f"rolling_min_{window}"] = (
            shifted
            .groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda x: x.rolling(window, min_periods=1).min())
        )
        # Rolling max
        df[f"rolling_max_{window}"] = (
            shifted
            .groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda x: x.rolling(window, min_periods=1).max())
        )
    return df


def create_expanding_mean(df, group_cols, target_col):
    """Expanding mean per group (leak-free target encoding)."""
    shifted = df.groupby(group_cols)[target_col].shift(1)
    group_key = df[group_cols].apply(tuple, axis=1)
    df["group_expanding_mean"] = (
        shifted
        .groupby(group_key)
        .transform(lambda x: x.expanding(min_periods=1).mean())
    )
    df["group_expanding_std"] = (
        shifted
        .groupby(group_key)
        .transform(lambda x: x.expanding(min_periods=1).std())
    )
    return df


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Full feature pipeline with categorical encoding
    """
    date_col = config["data"]["date_column"]
    target_col = config["data"]["target"]
    group_cols = ["product_id", "store_id"]

    # ----------------------
    # TIME FEATURES
    # ----------------------
    df = create_time_features(df, date_col)

    # ----------------------
    # LAG FEATURES
    # ----------------------
    lags = config["features"].get("lag_days", [1, 7, 14, 30])
    df = create_lag_features(
        df,
        group_cols=group_cols,
        target_col=target_col,
        lags=lags,
    )

    # ----------------------
    # ROLLING FEATURES (fixed per-group)
    # ----------------------
    windows = config["features"].get("rolling_windows", [7, 14, 30])
    df = create_rolling_features(
        df,
        group_cols=group_cols,
        target_col=target_col,
        windows=windows,
    )

    # ----------------------
    # EXPANDING MEAN (target encoding)
    # ----------------------
    df = create_expanding_mean(df, group_cols, target_col)

    # ----------------------
    # LAG RATIOS
    # ----------------------
    if "lag_1" in df.columns and "lag_7" in df.columns:
        df["lag_1_over_7"] = df["lag_1"] / (df["lag_7"] + 1e-8)
    if "lag_1" in df.columns and "lag_14" in df.columns:
        df["lag_1_over_14"] = df["lag_1"] / (df["lag_14"] + 1e-8)

    # ----------------------
    # DEMAND DEVIATION FROM GROUP MEAN
    # ----------------------
    if "lag_1" in df.columns:
        df["lag1_minus_group_mean"] = df["lag_1"] - df["group_expanding_mean"]

    # ----------------------
    # PRICE DYNAMICS
    # ----------------------
    df["price_change"] = df.groupby(group_cols)["price"].diff()
    df["price_change_pct"] = (
        df.groupby(group_cols)["price"].pct_change() * 100
    )

    # Price interaction with demand
    df["price_x_promo"] = df["price"] * df.get("promotion_flag", 0)

    # Price relative to group mean
    df["price_ratio_to_mean"] = df["price"] / (
        df.groupby(group_cols)["price"].transform("mean") + 1e-8
    )

    # ----------------------
    # DEMAND TREND
    # ----------------------
    shifted = df.groupby(group_cols)[target_col].shift(1)
    group_key = df[group_cols].apply(tuple, axis=1)

    df["trend_7"] = shifted.groupby(group_key).transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["trend_14"] = shifted.groupby(group_key).transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    df["trend_28"] = shifted.groupby(group_key).transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )

    # Trend momentum
    df["momentum_7_14"] = df["trend_7"] - df["trend_14"]
    df["momentum_7_28"] = df["trend_7"] - df["trend_28"]

    # ----------------------
    # PROMOTIONAL IMPACT
    # ----------------------
    if "promotion_flag" in df.columns:
        df["promo_lag_1"] = df.groupby(group_cols)["promotion_flag"].shift(1)
        promo_shifted = df.groupby(group_cols)["promotion_flag"].shift(1)
        df["promo_cumsum_7"] = (
            promo_shifted
            .groupby(group_key)
            .transform(lambda x: x.rolling(7, min_periods=1).sum())
        )

    # ----------------------
    # SEASONAL INTERACTION
    # ----------------------
    df["month_x_demand_rolling"] = df["month"] * df.get("trend_7", 1)
    df["dow_x_demand_rolling"] = df["day_of_week"] * df.get("trend_7", 1)

    # ----------------------
    # ECONOMIC INDEX FEATURES
    # ----------------------
    if "economic_index" in df.columns:
        df["econ_x_price"] = df["economic_index"] * df["price"]
        df["econ_x_promo"] = df["economic_index"] * df.get("promotion_flag", 0)

    # ----------------------
    # FILL NaN for new features
    # ----------------------
    df = df.fillna(0)

    # ----------------------
    # CATEGORICAL INTERACTIONS
    # ----------------------
    if "product_id" in df.columns:
        df["product_store_interaction"] = df["product_id"] * df["store_id"]

    if "category_id" in df.columns:
        df["category_store_interaction"] = df["category_id"] * df["store_id"]

    # ----------------------
    # FINAL CLEANUP
    # ----------------------
    df = df[df[target_col].notna()]

    return df
# Feature engineering
import pandas as pd

# ------------------------------- Function: Generate Features -----------------------------
def create_features(df):
    """
    Creates lag features and moving averages for the dataset.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing historical data.

    Returns:
        pd.DataFrame: DataFrame with additional feature columns.
    """
    df["price_diff"] = df["close"] - df["open"]
    df["5h_ma"] = df["close"].rolling(window=5).mean()  # 5-hour moving average
    df["10h_ma"] = df["close"].rolling(window=10).mean()  # 10-hour moving average
    df["lag_1"] = df["close"].shift(1)  # Previous hour close
    df["lag_2"] = df["close"].shift(2)  # 2-hour lag close
    return df.dropna()

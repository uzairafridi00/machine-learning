import requests
import pandas as pd
import os

# ------------------------------- Function: Fetch Historical Data -----------------------------
def get_historical_data(symbol="BTCUSDT", interval="1h", limit=500):
    """
    Fetches historical data from Binance API.
    
    Parameters:
        symbol (str): The trading pair symbol (e.g., "BTCUSDT").
        interval (str): The time interval for candles (e.g., "1h").
        limit (int): The number of candles to fetch.

    Returns:
        pd.DataFrame: DataFrame containing historical data.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    response = requests.get(url, params=params)
    data = response.json()
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df[["open_time", "open", "high", "low", "close", "volume"]]

# ------------------------------- Function: Fetch Live Price -----------------------------
def get_live_price(symbol="BTCUSDT"):
    """
    Fetches the current live price from Binance API.
    
    Parameters:
        symbol (str): The trading pair symbol (e.g., "BTCUSDT").

    Returns:
        float: The current price, or None if an error occurs.
    """
    url = "https://api.binance.com/api/v3/ticker/price"
    try:
        response = requests.get(url, params={"symbol": symbol})
        data = response.json()

        # Check if response is valid
        if "price" in data:
            live_price = float(data["price"])
            return live_price
        else:
            raise ValueError("Price data not found in response.")
    except Exception as e:
        print(f"Error fetching live price: {e}")
        return None


# ------------------------------- Function: Save Data to CSV -----------------------------
def save_to_csv(df, filename="bitcoin_data.csv"):
    """
    Saves the DataFrame to a CSV file. Appends if the file already exists.
    
    Parameters:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Name of the CSV file.
    """
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode="a", header=False, index=False)

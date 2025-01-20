import requests

# def get_live_bitcoin_price():
#     url = "https://api.binance.com/api/v3/ticker/price"
#     params = {"symbol": "BTCUSDT"}
#     response = requests.get(url, params=params)
#     data = response.json()
#     return float(data["price"])

# live_price = get_live_bitcoin_price()
# print(f"Live Bitcoin Price: ${live_price}")

import pandas as pd

def get_historical_bitcoin_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",  # 1 day intervals
        "limit": 365       # Last 365 days
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=columns)

    # Convert relevant columns to numeric
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["open_time", "open", "high", "low", "close", "volume"]]

historical_data = get_historical_bitcoin_data()
print(historical_data.head())

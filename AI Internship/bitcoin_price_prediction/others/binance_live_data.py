import requests
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

# ------------------------------- Function: Fetch Live Price -----------------------------
def get_live_price(symbol="BTCUSDT"):
    """
    Fetches the current live price from Binance API.
    
    Parameters:
        symbol (str): The trading pair symbol (e.g., "BTCUSDT").

    Returns:
        float: The current price.
    """
    url = "https://api.binance.com/api/v3/ticker/price"
    response = requests.get(url, params={"symbol": symbol})
    live_price = float(response.json()["price"])
    return live_price

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

# ------------------------------- Step 1: Load Historical Data -----------------------------
historical_data = get_historical_data()
save_to_csv(historical_data)  # Save the initial historical data to CSV

# ------------------------------- Step 2: Feature Engineering -----------------------------
data_with_features = create_features(historical_data)

# ------------------------------- Step 3: Define Features and Target -----------------------------
X = data_with_features[["price_diff", "5h_ma", "10h_ma", "lag_1", "lag_2", "volume"]]
y = data_with_features["close"]

# ------------------------------- Step 4: Train-Test Split -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------- Step 5: Train XGBoost Model -----------------------------
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ------------------------------- Step 6: Evaluate the Model -----------------------------
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-Squared:", r2_score(y_test, y_pred))

# ------------------------------- Continuous Live Prediction Loop -----------------------------
print("Starting live prediction...")
while True:
    try:
        # ------------------------------- Fetch Live Data -----------------------------
        live_price = get_live_price()
        current_time = pd.Timestamp.now()
        
        # Create a new row for live data
        new_row = {
            "open_time": current_time,
            "open": live_price,
            "high": live_price,
            "low": live_price,
            "close": live_price,
            "volume": 0.0,  # Placeholder volume
        }
        live_data = pd.DataFrame([new_row])
        
        # Append live data to historical data
        historical_data = pd.concat([historical_data, live_data], ignore_index=True)
        save_to_csv(live_data)  # Save the live data to CSV
        
        # ------------------------------- Update Features -----------------------------
        data_with_features = create_features(historical_data)
        
        # Prepare the latest features for prediction
        latest_features = data_with_features[["price_diff", "5h_ma", "10h_ma", "lag_1", "lag_2", "volume"]].iloc[-1]
        prediction = model.predict([latest_features])
        
        # ------------------------------- Print Prediction -----------------------------
        print(f"Time: {current_time}, Predicted Next Price: ${prediction[0]:.2f}")
        
        # Wait for 1 minute before fetching new data
        time.sleep(60)
    
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)  # Retry after 1 minute if an error occurs
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# ------------------------------- Function: Feature Engineering -----------------------------
def create_features(df):
    """
    Creates lag features and moving averages for the dataset.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing historical data.

    Returns:
        pd.DataFrame: DataFrame with additional feature columns.
    """
    df["price_diff"] = df["close"] - df["open"]
    df["5h_ma"] = df["close"].rolling(window=5).mean()
    df["10h_ma"] = df["close"].rolling(window=10).mean()
    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    return df.dropna()

# ------------------------------- Function: Train Model -----------------------------
def train_model(data_with_features):
    """
    Train the XGBoost model on the provided dataset.
    
    Parameters:
        data_with_features (pd.DataFrame): Data with features and target column.

    Returns:
        model: Trained XGBoost model.
    """
    X = data_with_features[["price_diff", "5h_ma", "10h_ma", "lag_1", "lag_2", "volume"]]
    y = data_with_features["close"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R-Squared:", r2_score(y_test, y_pred))

    return model

# ------------------------------- Function: Make Prediction -----------------------------
def predict_next_price(model, live_data):
    """
    Predict the next price using the trained model and live data.

    Parameters:
        model: The trained machine learning model.
        live_data (pd.DataFrame): Latest live data.

    Returns:
        float: Predicted price.
    """
    live_data["price_diff"] = live_data["close"] - live_data["open"]
    live_data["5h_ma"] = live_data["close"].rolling(window=5).mean().iloc
    live_data["10h_ma"] = live_data["close"].rolling(window=10).mean().iloc[-1]
    live_data["lag_1"] = live_data["close"].shift(1).iloc[-1]
    live_data["lag_2"] = live_data["close"].shift(2).iloc[-1]

    # Prepare the features for prediction
    latest_features = live_data[["price_diff", "5h_ma", "10h_ma", "lag_1", "lag_2", "volume"]].iloc[-1:]
    
    # Predict the next price
    prediction = model.predict(latest_features)
    return prediction[0]

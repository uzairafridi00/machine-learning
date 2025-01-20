import streamlit as st
import time
import pandas as pd
from fetch_data import get_historical_data, get_live_price, save_to_csv
from model import create_features, train_model, predict_next_price

# ------------------------------- Streamlit UI -----------------------------
st.title('Bitcoin Live Price Prediction')

# Dropdown for selecting time interval (1, 5, 15, 30, 60 minutes)
interval = st.selectbox("Select time interval for prediction:", [1, 5, 15, 30, 60])

# Load initial data and model
historical_data = get_historical_data(interval=f"{interval}m")
save_to_csv(historical_data)  # Save the initial historical data to CSV

# Create features for the data
data_with_features = create_features(historical_data)

# Train the XGBoost model
model = train_model(data_with_features)

# ------------------------------- Continuous Live Prediction -----------------------------
st.subheader("Live Prediction")

while True:
    try:
        # Fetch the latest live price
        live_price = get_live_price(interval)
        
        if live_price is None:
            st.write("Error: Unable to fetch live price.")
            time.sleep(60)  # Retry after 1 minute
            continue

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

        # Append live data to historical data and save to CSV
        historical_data = pd.concat([historical_data, live_data], ignore_index=True)
        save_to_csv(live_data)

        # Update features and make a prediction
        data_with_features = create_features(historical_data)
        prediction = predict_next_price(model, data_with_features)

        # Display the predicted next price
        st.write(f"Time: {current_time}, Predicted Next Price: ${prediction:.2f}")

        # Wait for the selected interval before fetching new data
        time.sleep(interval * 60)
    
    except Exception as e:
        st.write(f"Error: {e}")
        time.sleep(60)  # Retry after 1 minute if an error occurs

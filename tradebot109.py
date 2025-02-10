import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Setup logging
logging.basicConfig(
    filename="crypto_trading.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)

def log_message(action, result):
    logging.info(f"action: {action} - result: {result}")
    print(f"{action} - {result}")  # Print to console for immediate feedback

SYMBOL = "BTC-USD"
INTERVAL = "5m"  # Example: "1m", "5m", "1h"
LIMIT = 1000
THRESHOLD = 0.0001  # Adjusted to allow small profits for buy/sell decision

# Fetch historical data
def fetch_yfinance_data(symbol, interval, period="30d"):
    action = f"Fetching data for {symbol} with interval {interval}"
    try:
        df = yf.download(tickers=symbol, interval=interval, period=period)
        df.reset_index(inplace=True)
        result = f"Successfully fetched {len(df)} rows."
        log_message(action, result)
        return df
    except Exception as e:
        result = f"Error fetching data: {e}"
        log_message(action, result)
        return None

# Train a model
def train_model(df):
    action = "Training model"
    try:
        X = df[["Open", "High", "Low", "Close", "Volume"]]
        y = df["Close"].shift(-1).ffill().values.ravel()  # Flatten to 1D

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        result = "Model training successful."
        log_message(action, result)
        return model
    except Exception as e:
        result = f"Model training failed: {e}"
        log_message(action, result)
        return None

# Make a decision
def make_decision(latest_price, predicted_price, threshold):
    action = "Making trade decision"
    try:
        profit_margin = predicted_price - latest_price
        if profit_margin > 0 and profit_margin / latest_price >= threshold:
            result = "BUY"
        elif profit_margin < 0 and abs(profit_margin) / latest_price >= threshold:
            result = "SELL"
        else:
            result = "HOLD"
        log_message(action, f"Latest Price: {latest_price}, Predicted Price: {predicted_price}, Profit Margin: {profit_margin:.2f} - Decision: {result}")
        return result
    except Exception as e:
        result = f"Decision-making failed: {e}"
        log_message(action, result)
        return "HOLD"

# Execute a trade
def execute_trade(decision, symbol, amount):
    action = f"Executing trade: {decision}"
    try:
        result = f"Symbol: {symbol}, Amount: {amount}"
        log_message(action, result)
    except Exception as e:
        result = f"Trade execution failed: {e}"
        log_message(action, result)

# Live trading simulation
def live_trading():
    action = "Starting live trading"
    log_message(action, "Initialized")
    df = fetch_yfinance_data(SYMBOL, INTERVAL, period="30d")
    if df is None or df.empty:
        log_message(action, "No data fetched. Exiting.")
        return

    model = train_model(df)
    if model is None:
        log_message(action, "Model training failed. Exiting.")
        return

    latest_price = df["Close"].iloc[-1].item()  # Ensure scalar value
    features = df[["Open", "High", "Low", "Close", "Volume"]].iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(features)[0]  # Ensure scalar value

    decision = make_decision(latest_price, predicted_price, THRESHOLD)
    execute_trade(decision, SYMBOL, 0.001)
    log_message(action, "Completed")
    print("Live trading session completed.")  # Notify user of completion

# Run the script
if __name__ == "__main__":
    live_trading()

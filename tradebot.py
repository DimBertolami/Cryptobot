#!/usr/bin/env python3
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
    datefmt="%d-%m-%Y %H:%M:%S"
)

SYMBOL = "BTC-USD"
INTERVAL = "5m"  # Example: "1m", "5m", "1h"
LIMIT = 1000
THRESHOLD = 0.01  # 1% threshold for buy/sell decision

# Fetch historical data
def fetch_yfinance_data(symbol, interval, period="30d"):
    print(f"Fetching data for {symbol} with interval {interval}...")
    try:
        df = yf.download(tickers=symbol, interval=interval, period=period)
        df.reset_index(inplace=True)
        print(f"Successfully fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Train a model
def train_model(df):
    X = df[["Open", "High", "Low", "Close", "Volume"]]
    # y = df["Close"].shift(-1).fillna(method="ffill").values.ravel()  # Flatten to 1D
    y = df["Close"].shift(-1).ffill().values.ravel()  # Flatten to 1D

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Make a decision
def make_decision(latest_price, predicted_price, threshold):
    print(f"Latest Price: {latest_price}, Predicted Price: {predicted_price}")
    if predicted_price > latest_price * (1 + threshold):
        return "BUY"
    elif predicted_price < latest_price * (1 - threshold):
        return "SELL"
    else:
        return "HOLD"

# Execute a trade
def execute_trade(decision, symbol, amount):
    print(f"Decision: {decision}, Symbol: {symbol}, Amount: {amount}")

# Live trading simulation
def live_trading():
    df = fetch_yfinance_data(SYMBOL, INTERVAL, period="30d")
    if df is None or df.empty:
        print("No data fetched, exiting.")
        return

    model = train_model(df)
    print("Model trained.")

    # Fetch the latest price and prepare features
    latest_price = df["Close"].iloc[-1].item()  # Ensure scalar value
    features = df[["Open", "High", "Low", "Close", "Volume"]].iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(features)[0]  # Ensure scalar value

    decision = make_decision(latest_price, predicted_price, THRESHOLD)
    execute_trade(decision, SYMBOL, 0.001)

# Run the script
if __name__ == "__main__":
    live_trading()

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import time
import requests

'''
UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names
'''

# Initialize counter
# Global variable to track start time
start_time = time.time()
time.sleep(3)
elapsed_time = time.time() - start_time

#                   ##############################################################
duration = 23       # 5 minutes in seconds                                       #
INTERVAL = "5m"     # 1m, 5m, 30m, 1h, 10h, 8d, 30d, ...                         #
COUNTER = 0         # progress (^.^) (counter) in seconds..                      #
SYMBOL = "BTC-USD"  # cryptoID /   \ currency                                    #
THRESHOLD = 0.0001  # Adjusted  ||| to allow small profits for buy/sell decision #
#                   ##############################################################

# Setup logging
logging.basicConfig(
    filename="crypto.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)

def log_message(action, result):
    logging.info(f"action: {action} - result: {result}")
    print(f"{action} - {result}")  # Print to console for immediate feedback

# Fetch historical data from Yahoo Finance
def fetch_yfinance_data(symbol, interval, period="30d"):
    action = f"Requesting {symbol} with interval {interval}"
    try:
        df = yf.download(tickers=symbol, interval=interval, period=period)
        df.reset_index(inplace=True)
        result = f"received {len(df)}."
        log_message(action, result)
        return df
    except Exception as e:
        result = f"Error (yfinance): {e}"
        log_message(action, result)
        return None

# Fetch historical data from CoinGecko
def fetch_coingecko_data(symbol, currency="usd", days="30"):
    action = f"Requesting data from CoinGecko for {symbol}"
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency={currency}&days={days}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        prices = data.get("prices", [])
        if not prices:
            result = "No data from CoinGecko."
            log_message(action, result)
            return None

        # Convert prices to a DataFrame
        df = pd.DataFrame(prices, columns=["timestamp", "Close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Add placeholder columns for Open, High, Low, and Volume
        df["Open"] = df["Close"]  # Assuming no significant difference
        df["High"] = df["Close"] * 1.01  # Simulated 1% price variation
        df["Low"] = df["Close"] * 0.99  # Simulated 1% price variation
        df["Volume"] = np.random.uniform(1, 100, len(df))  # Simulated volume

        result = f"Successfully received {len(df)} from CoinGecko."
        log_message(action, result)
        return df
    except Exception as e:
        result = f"Error (CoinGecko): {e}"
        log_message(action, result)
        return None

# Fetch historical data from Yahoo Finance in chunks
def fetch_yfinance_data_in_chunks(symbol, interval, total_days):
    action = f"Requesting {symbol} from yfinance in API friendly interval {interval}"
    try:
        chunk_days = 8  # Limit for 1-minute data per request
        all_data = []
        end_date = pd.Timestamp.now()

        for _ in range(0, total_days, chunk_days):
            start_date = end_date - pd.Timedelta(days=chunk_days)
            df = yf.download(
                tickers=symbol,
                interval=interval,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d")
            )
            if not df.empty:
                all_data.append(df)
                result = f"Received {len(df)} for period: {start_date} to {end_date}."
                log_message(action, result)
            else:
                result = f"No data for {start_date} to {end_date}."
                log_message(action, result)

            end_date = start_date  # Move to the next chunk

        # Combine all chunks
        if all_data:
            combined_df = pd.concat(all_data)
            combined_df.reset_index(inplace=True)
            result = f"received {len(combined_df)} from yfinance..."
            log_message(action, result)
            return combined_df
        else:
            result = "No data fetched in any chunk."
            log_message(action, result)
            return None
    except Exception as e:
        result = f"Error (yfinance Chuncks): {e}"
        log_message(action, result)
        return None

# Train a Random Forest model
def train_random_forest(df):
    action = "Training Random Forest model"
    try:
        X = df[["Open", "High", "Low", "Close", "Volume"]]
        y = df["Close"].shift(-1).ffill().values.ravel()  # Flatten to 1D

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        result = "successful."
        log_message(action, result)
        return model
    except Exception as e:
        result = f"failed: {e}"
        log_message(action, result)
        return None

# Train a Linear Regression model
def train_linear_regression(df):
    action = "Training Linear Regression model"
    try:
        X = df[["Open", "High", "Low", "Close", "Volume"]]
        y = df["Close"].shift(-1).ffill().values.ravel()  # Flatten to 1D

        model = LinearRegression()
        model.fit(X, y)
        result = "successful."
        log_message(action, result)
        return model
    except Exception as e:
        result = f"failed: {e}"
        log_message(action, result)
        return None

# Train a Support Vector Regressor (SVR) model
def train_svr(df):
    action = "Training Support Vector Regressor model"
    try:
        X = df[["Open", "High", "Low", "Close", "Volume"]]
        y = df["Close"].shift(-1).ffill().values.ravel()  # Flatten to 1D

        model = SVR()
        model.fit(X, y)
        result = "successful."
        log_message(action, result)
        return model
    except Exception as e:
        result = f"failed: {e}"
        log_message(action, result)
        return None

# Train a K-Nearest Neighbors (KNN) Regressor model
def train_knn(df):
    action = "Training K-Nearest Neighbors model"
    try:
        X = df[["Open", "High", "Low", "Close", "Volume"]]
        y = df["Close"].shift(-1).ffill().values.ravel()  # Flatten to 1D

        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X, y)
        result = "successful."
        log_message(action, result)
        return model
    except Exception as e:
        result = f"failed: {e}"
        log_message(action, result)
        return None

# Train a Gradient Boosting Regressor model
def train_gradient_boosting(df):
    action = "Training Gradient Boosting Regressor model"
    try:
        X = df[["Open", "High", "Low", "Close", "Volume"]]
        y = df["Close"].shift(-1).ffill().values.ravel()  # Flatten to 1D

        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        result = "successful."
        log_message(action, result)
        return model
    except Exception as e:
        result = f"failed: {e}"
        log_message(action, result)
        return None

# Make a decision
def make_decision(latest_price, predicted_price, threshold):
    action = "Making trade decision"
    try:
        # Ensure latest_price and predicted_price are scalar
        if isinstance(latest_price, pd.Series):
            latest_price = latest_price.iloc[0]  # Get scalar value from Series
        if isinstance(predicted_price, pd.Series):
            predicted_price = predicted_price.iloc[0]  # Get scalar value from Series

        # Convert to float if necessary
        latest_price = float(latest_price)
        predicted_price = float(predicted_price)

        # Check for NaN values
        if pd.isna(latest_price) or pd.isna(predicted_price):
            log_message(action, "Latest price or predicted price is NaN. Decision: HOLD")
            return "HOLD"

        profit_margin = predicted_price - latest_price
        if profit_margin > 0 and profit_margin / latest_price > threshold:
            result = "BUY"
        elif profit_margin < 0 and abs(profit_margin) / latest_price > threshold:
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
def live_trading(use_random_forest=True, source="yfinance"):
    action = "Starting live trading"
    log_message(action, "Initialized")

    # Fetch data based on the source
    if source == "yfinance":
        log_message(action, "Attempting to fetch data from yfinance in chunks")
        df = fetch_yfinance_data_in_chunks(SYMBOL, INTERVAL, total_days=30)
      # log_message(action, "Attempting to get data from yfinance")
      # df = fetch_yfinance_data(SYMBOL, INTERVAL, period="30d")
    elif source == "coingecko":
        log_message(action, "Attempting to get data from coingecko")
        df = fetch_coingecko_data("bitcoin", days="30")
    else:
        log_message(action, "Invalid data source. Exiting.")
        return

    if df is None or df.empty:
        log_message(action, "No data fetched. Exiting.")
        return
    else:
        log_message(action, f"Data from {source}. Rows: {len(df)}")

    # Train the model
    log_message(action, "Training model...")

    while time.time() - start_time < duration:
        # Train models
        rf_model  = train_random_forest(df)
        lr_model  = train_linear_regression(df)
        svr_model = train_svr(df)
        knn_model = train_knn(df)
        gb_model  = train_gradient_boosting(df)
        time.sleep(1)
        elapsed_time = time.time() - start_time
        print(f"time elapsed: {elapsed_time}")

    #choose a model
    model = train_random_forest(df) if use_random_forest else train_linear_regression(df)

    if model is None:
        log_message(action, "Model training failed. Exiting.")
        return

    log_message(action, "training completed. now make a decision.")

    # Decision-making loop (simulate live trading)
    for i in range(len(df) - 1):  # Loop through historical data
        try:
            latest_price = df["Close"].iloc[i].item() 
            #latest_price = df["Close"].iloc[i]
            features = df[["Open", "High", "Low", "Close", "Volume"]].iloc[i].values.reshape(1, -1)
            predicted_price = model.predict(features)[0]  # Predicted price as scalar
            profit_margin = predicted_price - latest_price
            decision = make_decision(latest_price, predicted_price, THRESHOLD)
            print(f"Latest Price: ${latest_price:.2f}")
            print(f"Predicted Price: ${predicted_price:.2f}")
            print(f"Profit Margin: {profit_margin:.2f} - Decision: {decision}")
            elapsed_time = time.time() - start_time
            print(f"elapsed time: {elapsed_time}")
           # Delay between iterations
            time.sleep(2)

        except Exception as e:
            log_message(action, f"Error in trading loop: {e}")
            continue

# choose prediction model
# 	- set use_random_forest 	to True 	to use random_forest
# 	- use_lr_model 			to True		to use lr_model
# 	- use_svr_model 		to True 	to use svr_model
# 	- use_knn_model 		to True 	to use knn_model
# 	- use_gb_model 			to True 	to use gb_model

if __name__ == "__main__":
    # Start live trading
    live_trading(use_random_forest=True, source="yfinance")  # Change to "coingecko" if yfinance isn't working

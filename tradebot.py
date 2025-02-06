#!/usr/bin/env python3
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.cluster import KMeans
from datetime import datetime
import requests
import pandas as pd
import logging
import numpy as np
import time
import traceback

symbol = "bitcoin"
currency = "usd"
limit = 100
interval_seconds = 300


logging.basicConfig(
    filename="crypto_trading.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)

def cluster_prices(prices, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    prices['Cluster'] = kmeans.fit_predict(prices[['price']])
    return prices

# ML Models and logic
def predict_with_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_with_logistic_regression(X_train, y_train, X_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def predict_with__decision_tree(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

def predict_with_linear_regression(prices):
    # Prepare data
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices['price'].values
    model = LinearRegression()
    model.fit(X, y)
    # Predict next price
    next_time = [[len(prices)]]
    next_price = model.predict(next_time)
    return next_price

def moving_average_strategy(prices, window=10):
    # Calculate moving average
    prices['MA'] = prices['price'].rolling(window=window).mean()
    # Signal when price crosses above/below moving average
    prices['Signal'] = prices['price'] > prices['MA']
    return prices

def knn_classifier(X, y, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    return model

def fetch_coingecko_historical_data(crypto_id="bitcoin", vs_currency="usd", days="30"):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    response = requests.get(url, params=params)
    return response.json()

def fetch_binance_historical_data(symbol="BTCUSD", interval="1d", limit=30):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    return response.json()

def fetch_coingecko_data(crypto_id="bitcoin", vs_currency="usd"):
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": crypto_id,  # Cryptocurrency ID (e.g., 'bitcoin', 'ethereum')
        "vs_currencies": vs_currency  # Target currency (e.g., 'usd', 'eur')
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        print(f"CoinGecko: {crypto_id.capitalize()} price in {vs_currency.upper()}: {data[crypto_id][vs_currency]}")
    except Exception as e:
        print(f"Error fetching data from CoinGecko: {e}")

# CoinMarketCap API Function
def fetch_coinmarketcap_data(api_key, crypto_symbol="BTC"):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": api_key,
    }
    params = {
        "symbol": crypto_symbol  # Cryptocurrency symbol (e.g., 'BTC', 'ETH')
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        price = data["data"][crypto_symbol]["quote"]["USD"]["price"]
        return f"CoinMarketCap: {crypto_symbol} price in USD: {price}"
    except Exception as e:
        return f"Error fetching data from CoinMarketCap: {e}"

# Alpaca API Function
def fetch_alpaca_crypto_data(api_key, api_secret, crypto_symbol="BTC/USD"):
    base_url = "https://paper-api.alpaca.markets"  # Use the paper trading URL for testing

    # Initialize the Alpaca REST API
    api = REST(api_key, api_secret, base_url)
    try:
        # Fetch the latest trade data for the given crypto pair
        trade = api.get_latest_trade(crypto_symbol)
        return f"Alpaca: Latest {crypto_symbol} price: {trade.price}"
    except Exception as e:
        return f"Error fetching data from Alpaca: {e}"

# Fetch Data
def fetch_data():
    logging.info("Fetching historical and live data...")
    try:
        historical_data = fetch_coingecko_historical_data("bitcoin", "eur", "90")
        logging.info(f"Legacy price data: {historical_data}")
        live_data = fetch_coingecko_data(crypto_id="bitcoin", vs_currency="usd")
        logging.info(f"price  data today: {live_data}")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise
    return historical_data, live_data

# Preprocess Data
def preprocess_data(historical_data):
    logging.info("Preprocessing data...")
    try:
        df = pd.DataFrame(historical_data)
        df['moving_avg'] = moving_average_strategy(df['prices'], window=10)
        df['clusters'] = cluster_prices(df['prices'], n_clusters=3)
        logging.info("using moving average and clustering to preprocess the data.")
    except Exception as e:
        logging.error(f"Error during data preparation: {e}")
        raise
    return df

# Train Models
def train_models(df):
    logging.info("Learning...")
    try:
        lstm_model = predict_with_lstm(input_shape=(len(df['prices']), 1))
        logistic_model = predict_with_logistic_regression(df[['prices']], df['clusters'], df[['prices']])
        knn_model = knn_classifier(df[['prices']], df['clusters'], k=5)
        logging.info("Model ready.")
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise
    return lstm_model, logistic_model, knn_model

# Make Predictions
def make_predictions(models, live_data):
    logging.info("Making predictions...")
    try:
        lstm_model, logistic_model, knn_model = models
        lstm_prediction = lstm_model.predict(live_data)
        logging.info(f"Prediction completed: {lstm_prediction}")
    except Exception as e:
        logging.error(f"Error during predictions: {e}")
        raise
    return lstm_prediction

# Decide based on prediction
def decision_making(prediction, live_data):
    logging.info("Making trading decision...")
    try:
        decision = make_decision(prediction, live_data)
        logging.info(f"Decision made: {decision}")
    except Exception as e:
        logging.error(f"Error during decision making: {e}")
        raise
    return decision

# Execute Trades
def execute_trades(decision):
    logging.info("Executing trade...")
    try:
        if decision == "buy":
            live_trading()
            logging.info("Buying...")
        elif decision == "sell":
            live_trading()
            logging.info("Selling...")
        else:
            logging.info("Holding...")
    except Exception as e:
        logging.error(f"Error during trade execution: {e}")
        raise

# Main Function
def main():
    logging.info("Requesting data...")
    try:
        historical_data, live_data = fetch_data()
        print(f"historical data:  {historical_data}")
        print(f"live data:  {live_data}")
        df = preprocess_data(historical_data)
        print(f"data preprocessing completed")
        models = train_models(df)
        print(f"finished training models")
        prediction = make_predictions(models, live_data)
        print(f"autogenerated prediction:  {prediction}")
        decision = decision_making(prediction, live_data)
        print(f"based on prediction decision was made: {decision}")
        execute_trades(decision)
        print(f"the actual buy/sell logic at the moment just notifies (as per design)")
        logging.info("bot is shutting down.")
    except Exception as e:
        logging.error(f"error: {e}")

if __name__ == "__main__":
    main()

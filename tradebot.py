import requests
import logging
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(
    filename="crypto_trading.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)
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
        df = preprocess_data(historical_data)
        models = train_models(df)
        prediction = make_predictions(models, live_data)
        decision = decision_making(prediction, live_data)
        execute_trades(decision)
        logging.info("Trading bot is terminating.")
    except Exception as e:
        logging.error(f"error: {e}")

if __name__ == "__main__":
    main()

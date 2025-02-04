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

# Fetch Data
def fetch_data():
    logging.info("Fetching historical and live data...")
    try:
        historical_data = fetch_coingecko_historical_data(symbol="bitcoin", currency="usd", days="90")
        live_data = fetch_coingecko_data(crypto_id="bitcoin", vs_currency="usd")
        logging.info(f"Historical  data:  {historical_data}")
        logging.info(f"Live data:  {live_data}")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise
    return historical_data, live_data

def preprocess_data(historical_data):
    logging.info("Prepping data...")
    try:
        df = pd.DataFrame(historical_data)
        df['moving_avg'] = moving_average_strategy(df['prices'], window=10)
        df['clusters'] = cluster_prices(df['prices'], n_clusters=3)
        logging.info("Data preparation completed successfully.")
    except Exception as e:
        logging.error(f"Error prepping  data: {e}")
        raise
    return df

def train_models(df):
    logging.info("Training models...")
    try:
        lstm_model = predict_with_lstm(input_shape=(len(df['prices']), 1))
        logistic_model = predict_with_logistic_regression(df[['prices']], df['clusters'], df[['prices']])
        knn_model = knn_classifier(df[['prices']], df['clusters'], k=5)
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
    return lstm_model, logistic_model, knn_model

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

def decision_making(prediction, live_data):
    logging.info("Making trading decision...")
    try:
        decision = make_decision(prediction, live_data)
        logging.info(f"Decision made: {decision}")
    except Exception as e:
        logging.error(f"Error during decision making: {e}")
        raise
    return decision

def execute_trades(decision):
    logging.info("Executing trade...")
    try:
        if decision == "buy":
            live_trading()
            logging.info("Executed BUY trade.")
        elif decision == "sell":
            live_trading()
            logging.info("Executed SELL trade.")
        else:
            logging.info("No trade executed. Decision was HOLD.")
    except Exception as e:
        logging.error(f"Error during trade execution: {e}")
        raise

def main():
    logging.info("Starting trading bot...")
    try:
        historical_data, live_data = fetch_data()
        df = preprocess_data(historical_data)
        models = train_models(df)
        prediction = make_predictions(models, live_data)
        decision = decision_making(prediction, live_data)
        execute_trades(decision)
        logging.info("Trading bot run completed.")
    except Exception as e:
        logging.error(f"Critical error in trading bot: {e}")

if __name__ == "__main__":
    main()

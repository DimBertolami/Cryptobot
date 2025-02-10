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
        logging.info("Data fetching completed successfully.")
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
        logging.info("Data preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise
    return df

# Train Models
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

# Make a Decision
def decision_making(prediction, live_data):
    logging.info("Making trading decision...")
    try:
        decision = make_decision(prediction, live_data)
        logging.info(f"Decision made: {decision}")
    except Exception as e:
        logging.error(f"Error during decision making: {e}")
        raise
    return decision

def live_trading(symbol="BTCUSDT", interval="1h", quantity=0.001, threshold=0.01):
    """
    Simulate live trading using Binance and a trained model.
    """
    # Fetch historical data
    df = fetch_binance_historical_data(symbol, interval)
    model = train_model(df)

    while True:
        # Fetch the latest price
        latest_price = float(client.get_symbol_ticker(symbol=symbol)["price"])

        # Predict next price
        features = np.array([[df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], latest_price, df["volume"].iloc[-1]]])
        predicted_price = model.predict(features)[0]

        # Make decision
        decision = make_decision(latest_price, predicted_price, threshold)
        print(f"Latest Price: {latest_price}, Predicted: {predicted_price}, Decision: {decision}")

        # Execute trade
        execute_trade(decision, symbol, quantity)

        # Wait for the next interval
        time.sleep(3600)  # 1 hour

# Execute Trades
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

# Main Function
def main():
    logging.info("Starting trading bot...")
    try:
        historical_data, live_data = fetch_data()
        print(f"historical data: {historical_data}")
        print(f"current data: {live_data}")
        df = preprocess_data(historical_data)
        print(f"df= {df}")
        models = train_models(df)
        print(f"models: {models}")
        prediction = make_predictions(models, live_data)
        print(f"prediction: {prediction}")
        decision = decision_making(prediction, live_data)
        print(f"decision: {decision}")
        execute_trades(decision)
        logging.info("Trading bot run completed.")
    except Exception as e:
        logging.error(f"Critical error in trading bot: {e}")

if __name__ == "__main__":
    main()

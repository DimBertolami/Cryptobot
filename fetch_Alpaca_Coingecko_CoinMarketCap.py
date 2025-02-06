'''

The fetch_coingecko_historical_data function fetches historical price data 
from CoinGecko for the specified cryptocurrency and currency (e.g., Bitcoin 
in USD) over the last n days.
It processes the JSON data into a pandas DataFrame with timestamp and price columns.

The train_linear_regression_model function creates a 
simple time index (X) and maps it to the price data (y).
It trains a LinearRegression model using scikit-learn.
It predicts the next price based on the model.

The plot_predictions function generates a plot of actual 
prices alongside the predicted trendline.

The script prints the predicted next price and shows a 
plot of the historical prices with the regression line.
'''
import os
import requests
from alpaca_trade_api.rest import REST
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fetch_coingecko_historical_data(crypto_id="bitcoin", vs_currency="usd", days="30"):
    """
    Fetch historical cryptocurrency data from CoinGecko.
    Args:
        crypto_id (str): The ID of the cryptocurrency (e.g., "bitcoin").
        vs_currency (str): The target currency (e.g., "usd").
        days (str): Number of past days for which to fetch data (e.g., "30" or "max").
    Returns:
        DataFrame: A pandas DataFrame with timestamps and prices.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract timestamps and prices
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    # Convert timestamp to readable datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def train_linear_regression_model(df):
    """
    Train a linear regression model on the given price data.
    Args:
        df (DataFrame): A DataFrame containing timestamps and prices.
    Returns:
        model (LinearRegression): The trained model.
        next_price (float): The predicted next price.
    """
    # Prepare the data for training
    df["time_index"] = np.arange(len(df))  # Create a time index for X values
    X = df["time_index"].values.reshape(-1, 1)  # Feature (time index)
    y = df["price"].values  # Target (prices)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next price (based on the next time index)
    next_time_index = [[len(df)]]  # The next time index
    next_price = model.predict(next_time_index)[0]

    return model, next_price

def plot_predictions(df, model):
    """
    Plot the historical prices and the linear regression line.
    Args:
        df (DataFrame): A DataFrame containing timestamps and prices.
        model (LinearRegression): The trained model.
    """
    # Prepare data for plotting
    df["time_index"] = np.arange(len(df))
    X = df["time_index"].values.reshape(-1, 1)
    y_pred = model.predict(X)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp"], df["price"], label="Actual Prices", marker="o")
    plt.plot(df["timestamp"], y_pred, label="Linear Regression Prediction", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Cryptocurrency Price Prediction with Linear Regression")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Fetch historical data
    crypto_id = "bitcoin"  # Change to your preferred cryptocurrency (e.g., "ethereum")
    vs_currency = "usd"
    days = "30"  # Fetch data for the past 30 days
    df = fetch_coingecko_historical_data(crypto_id, vs_currency, days)

    # Train the linear regression model
    model, next_price = train_linear_regression_model(df)

    # Print the predicted next price
    print(f"Predicted next price for {crypto_id.capitalize()} in {vs_currency.upper()}: {next_price:.2f}")

    # Plot the predictions
    plot_predictions(df, model)


# Run the Script
if __name__ == "__main__":
    main()

'''
1. Moving Average and Exponential Smoothing (Simple Time Series Techniques)
   is Moving averages smooth out price data to identify trends. 
   Combine them with thresholds to trigger buy/sell signals.
   No need for advanced ML libraries—just basic Python and libraries like pandas.
   Example Use Case: Detect bullish/bearish trends.
'''
def moving_average_strategy(prices, window=10):
    # Calculate moving average
    prices['MA'] = prices['price'].rolling(window=window).mean()
    # Signal when price crosses above/below moving average
    prices['Signal'] = prices['price'] > prices['MA']
    return prices
'''
2. Linear Regression
   It predicts future prices based on historical data trends. It's a supervised learning method that works well with numerical data.
   Libraries like scikit-learn make it simple to implement.
   Example Use Case: Predict the next price of Bitcoin.
'''

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
'''

3. Decision Trees
   Is a supervised ML algorithm that splits data into decision nodes. 
   Ideal for classifying price movements (e.g., UP, DOWN).
   Simple to train using scikit-learn.
   Example Use Case: Determine whether to buy, sell, or hold based on historical trends.
'''
def train_decision_tree(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model
'''
4. Logistic Regression
   Is a supervised ML model for binary classification. You can use it to predict whether the price will go UP or DOWN.
   Why It's Easy: Easy to implement with scikit-learn and great for binary decision-making.
   Example Use Case: Predict whether to buy/sell based on market indicators.
'''

def predict_with_logistic_regression(X_train, y_train, X_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions
'''
5. Random Forest
   Is an ensemble ML technique based on decision trees that averages predictions for better accuracy.
   scikit-learn simplifies implementation.
   Example Use Case: Predict price direction with historical features.
'''

def random_forest_classifier(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model
'''
6. k-Nearest Neighbors (k-NN)
   A simple classification technique that predicts based on the majority class of the k nearest data points.
   No model training—just distance calculations.
   Example Use Case: Classify price direction or clusters of high/low volatility.
'''

def knn_classifier(X, y, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    return model
'''
7. Long Short-Term Memory (LSTM) Neural Networks
   It's a type of recurrent neural network (RNN) designed for sequential data, like time series.
   It's a Bit Advanced! It requires libraries like TensorFlow or PyTorch, but highly effective for crypto price predictions.
   Example Use Case: Predict the next day's price based on historical trends.
'''

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
'''
8. Clustering (k-Means or DBSCAN)
   It's an unsupervised ML technique that groups data into clusters based on patterns.
   Why Easy: No labels are needed; useful for identifying market conditions.
   Example Use Case: Group periods of high volatility.
'''

def cluster_prices(prices, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    prices['Cluster'] = kmeans.fit_predict(prices[['price']])
    return prices

# CoinGecko API Function
def fetch_coingecko_data(crypto_id="bitcoin", vs_currency="usd"):
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": crypto_id,  # Cryptocurrency ID (e.g., 'bitcoin', 'ethereum')
        "vs_currencies": vs_currency  # Target currency (e.g., 'usd', 'eur')
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return f"CoinGecko: {crypto_id.capitalize()} price in {vs_currency.upper()}: {data[crypto_id][vs_currency]}"
    except Exception as e:
        return f"Error fetching data from CoinGecko: {e}"


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
def fetch_alpaca_crypto_data(ALP_key, ALP_secret, crypto_symbol="BTC/USD"):
    base_url = "https://paper-api.alpaca.markets"  # Use the paper trading URL for testing

    # Initialize the Alpaca REST API
    api = REST(os.getenv(ALP_key), os.getenv(ALP_secret), base_url)
    try:
        # Fetch the latest trade data for the given crypto pair
        trade = api.get_latest_trade(crypto_symbol)
        return f"Alpaca: Current {crypto_symbol} price: {trade.price}"
    except Exception as e:
        return f"Error fetching data from Alpaca: {e}"


# Main Function to Fetch Data from All APIs
def fetch_all_crypto_data():
    # API Keys (Replace with your actual keys)
    # export coinmarketcap_api_key="1758e18b-1744-4ad6-a2a9-908af2f33c8a"
    # export alpaca_api_key="AKR0KGK2HD95VYI0ZHUR"
    # export alpaca_api_secret="ZzyWUTFuMfppA6l0qiaRm5RxNgmiUAOhM6xJMhWu"
    # print(f"CMCap os.getenv(COINMARKETCAP_KEY}")
    # print(f"Alpaca Key: (os.getenv(ALP_KEY))")
    # print(f"Alpaca Secret: {os.getenv(ALP_SECRET)}")
    cg_key  = os.getenv("COINGECKO_KEY")
    CMC_key = os.getenv("COINMARKETCAP_KEY")
    ALP_secret = os.getenv("ALPACA_SECRET")
    ALP_key    = os.getenv("ALPACA_KEY")
    print(cg_key)
    print(CMC_key)
    print(ALP_secret)
    print(ALP_key)

    # Coin IDs and symbols
    crypto_id = "bitcoin"  # For CoinGecko
    crypto_symbol_cmc = "BTC"  # For CoinMarketCap
    crypto_symbol_alpaca = "BTC/USD"  # For Alpaca

    # Fetch data from all APIs
    coingecko_result = fetch_coingecko_data(crypto_id, "usd")
    coinmarketcap_result = fetch_coinmarketcap_data(CMC_key, crypto_symbol_cmc)
    alpaca_result = fetch_alpaca_crypto_data(ALP_key, ALP_secret, crypto_symbol_alpaca)

    # Print results
    print(coingecko_result)
    print(coinmarketcap_result)
    print(alpaca_result)


# Run the Script
if __name__ == "__main__":
    fetch_all_crypto_data()

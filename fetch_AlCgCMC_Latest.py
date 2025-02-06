import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


# Step 1: Fetch Historical Price Data from CoinGecko
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


# Step 2: Train a Linear Regression Model
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


# Step 3: Visualize the Results
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


# Step 4: Main Function to Fetch Data, Train Model, and Predict
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

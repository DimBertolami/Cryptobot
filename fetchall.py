'''	Author: Dimi Bertolami										   date: 16-03-2025
        ----------------------										   ----------------
1.0)    This bot came to life after realising that I suck at guessing which cryptocurrency is going to make me profit
1.1) 	install required packages
1.2) 	fetch historical price data and technical indicators.
2)   	Feature Engineering: Create features based on historical data and technical indicators (e.g., RSI, moving averages).
3)   	preprocess the datait for machine learning (model training for example normalize, generate technical indicators).
4)   	Train machine learning mode  (LSTM, Decision Trees, or RL agent).
5)   	Evaluate the models on a validation dataset or new data using metrics such as accuracy, precision, recall (for classification models), or profitability (for RL).
6)   	Use the model's predictions to implement a Buy/Hold/Sell strategy.
'''

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from ta.utils import dropna
from datetime import datetime
import matplotlib.pyplot as plt
import requests, talib, json, os
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from python_bitvavo_api.bitvavo import Bitvavo
from binance.client import Client as BinanceClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY")
BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET")
ALPACA_API_KEY =  os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')

'''
Decision Trees and Random Forests
Best for: Simpler models, fast training, interpretable outputs.
    • Why: Decision trees and random forests are powerful algorithms for classification tasks. You can frame the problem as a classification problem where you classify the market into different categories (Buy, Hold, Sell) based on past data.
    • How:
        ◦ Features: Use features like moving averages (e.g., SMA, EMA), RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), price volatility, and volume.
        ◦ Model: Random forests or gradient-boosted trees (e.g., XGBoost, LightGBM) are strong candidates for this problem because they handle non-linear relationships and can model complex interactions in the data.
        ◦ Example: Use historical OHLC (Open, High, Low, Close) data as input, with labels defined as Buy, Hold, and Sell based on price movement or other technical indicators.
Pros:
    • Easy to implement and interpret.
    • Can handle non-linear relationships.
    • Often performs well in classification tasks.
Cons:
    • Might not capture sequential dependencies as well as neural networks (e.g., long-term trends).
'''
def train_Random_Forest(n_estimators=100, test_size=0.2, shuffle=False, features = ['SMA14', 'EMA14', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']):
    # Select features and target, Split data into training and testing sets, Train Random Forest model, Make predictions and evaluate result
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, shuffle)
    rf = RandomForestClassifier(n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

'''
Recurrent Neural Networks (RNNs) and LSTM (Long Short-Term Memory) Networks
Best for: Capturing sequential data, time series prediction.
    • Why: RNNs, and especially LSTMs, are great for working with time-series data, such as price and volume in cryptocurrency markets, because they can capture long-term dependencies in the data.
    • How:
        ◦ Features: Use time-series data of price (OHLC), indicators (e.g., RSI, MACD, moving averages), and even external data like sentiment analysis.
        ◦ Model: Train an LSTM model where the input is a sequence of past prices and features, and the output is a prediction for the next action (Buy, Hold, Sell).
        ◦ Example: An LSTM network could learn to predict the next price movement based on past price movements and features, then use that prediction to make a buy/hold/sell decision.
Pros:
    • Ideal for sequential data and time-series forecasting.
    • Captures long-term dependencies in price movements.
Cons:
    • Requires a large amount of data and computation.
    • Difficult to interpret, less explainable than decision trees.
'''
def train_Recurrent_Neural_net_and_ltsm(feature_range=(0, 1),  lookback=60, units=50, return_sequences=True, activation='sigmoid', epochs=10, batch_size=32, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
# Use the last 60 prices to predict the next one
# Scale the data,  Prepare data for RNN,  Create sequences for RNN,  Reshape data for LSTM,  Build LSTM model, Convert probabilities to Buy/Sell
# Train the model, Make predictions,  Evaluate the model
    scaler = MinMaxScaler(feature_range)
    scaled_data = scaler.fit_transform(df[['close']])
    X = []
    y = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(df['target'].iloc[i])
        X = np.array(X)
        y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    model.add(LSTM(units, return_sequences, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units))
    model.add(Dense(units=1, activation='sigmoid'))  # Binary output (Buy/Sell)
    model.compile(optimizer, loss, metrics)
    model.fit(X, y, epochs, batch_size)
    y_pred = model.predict(X)
    y_pred = (y_pred > 0.5).astype(int)
    print(classification_report(y, y_pred))

'''
Convolutional Neural Networks (CNNs)
Best for: Feature extraction from time-series data, identifying patterns.
    • Why: CNNs are typically used for image data, but can also be applied to time-series data like financial data by treating the data as a 1D "image". CNNs can extract important features from the time-series, making them effective for detecting patterns and trends in market data.
    • How:
        ◦ Features: Use a window of historical data (e.g., the last 30 minutes, hours, or days) as input and apply CNN layers to learn spatial patterns from the data.
        ◦ Model: Combine CNNs with LSTMs to model both local patterns and long-term trends in cryptocurrency data. This hybrid model can be used to predict price movements and, subsequently, trading actions.
Pros:
    • Great for feature extraction and detecting patterns in the data.
    • Works well with raw time-series data (like price and volume).
Cons:
    • May require more data and tuning to perform well.
    • Less intuitive compared to decision trees.
'''
def train_Convolutional_Neural_Net(filters=64, kernel_size=2, activation='relu',units=1, activation2='sigmoid', epochs=10, batch_size=32):
# Reshape data for CNN, Build CNN model, Train the model, Make predictions, Evaluate the model
    X_cnn = X.reshape(X.shape[0], X.shape[1], 1)
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_cnn.shape[1], 1)))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(units=1, activation2='sigmoid'))
    model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_cnn.fit(X_cnn, y, epochs, batch_size)
    y_pred_cnn = model_cnn.predict(X_cnn)
    y_pred_cnn = (y_pred_cnn > 0.5).astype(int)
    print(classification_report(y, y_pred_cnn))

'''
Reinforcement Learning (RL)
Best for: Strategy optimization over time, dynamic decision-making.
    • Why: Reinforcement learning is a perfect fit for decision-making problems, where the agent (in this case, the model) interacts with an environment (the market) and learns to make decisions by maximizing cumulative rewards. In your case, the agent can learn when to buy, hold, or sell by simulating trades based on historical data and optimizing the strategy over time.
    • How:
        ◦ Features: Use technical indicators and price movements as state inputs. The agent gets rewarded when its actions lead to a profitable outcome.
        ◦ Model: A popular approach is using Q-learning, Deep Q Networks (DQN), or Proximal Policy Optimization (PPO) to define and optimize a trading strategy. The RL agent is trained using past data to maximize long-term profit or minimize loss.
Pros:
    • Optimizes decision-making over time.
    • Can adapt to changing market conditions.
    • Trains a model that learns to interact with the market directly.
0Cons:
    • Requires a lot of training data and computation.
    • Harder to implement and interpret.
    • Prone to overfitting without careful validation.
'''
def train_dqn(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close']])

    env = TradingEnvironment(scaled_data)
    state_size = env.lookback
    action_size = env.action_space

    agent = DQNAgent(state_size, action_size)

    epochs = 1000
    for e in range(epochs):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.replay()
        agent.update_target_model()
        print(f"Epoch {e}/{epochs}, Total Reward: {total_reward}")

'''
XGBoost (Gradient Boosting)
Best for: Structured data and boosting performance on classification tasks.
    • Why: XGBoost is an efficient implementation of gradient boosting and works well for predicting buy/hold/sell decisions in financial markets, as it can handle a variety of features like technical indicators, historical prices, and more.
    • How:
        ◦ Features: Use technical indicators, historical prices, and other relevant features to train a classification model.
        ◦ Model: Use XGBoost to classify market conditions as "Buy", "Hold", or "Sell" based on the extracted features.
Pros:
    • Highly efficient and fast for training.
    • Often performs well on tabular data (e.g., technical indicators).
Cons:
    • Doesn’t model time dependencies as well as RNNs or LSTMs.
    • Requires careful feature engineering.
'''
def train_xgboost_buy_hold_sell(df, features, target_col='target'):
    # Multi-class classification Three classes: Buy, Hold, Sell
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {' objective': 'multi:softmax', 'num_class': 3, 'max_depth': 6, 'learning_rate': 0.1, 'eval_metric': 'mlogloss'}
    model = xgb.train(params, dtrain, num_boost_round=100)
    y_pred = model.predict(dtest)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

# Calculate technical indicators
def calculate_indicators(df):
    df['SMA14'] = df['close'].rolling(window=14).mean()  								# Simple Moving Average
    df['EMA14'] = df['close'].ewm(span=14, adjust=False).mean()  							# Exponential Moving Average
    df['EMA'] = df['close'].ewm(span=14).mean()                       							# , adjust=False # Exponential Moving Average (14-period) technical indicator
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)  									# Relative Strength Index
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)  		# MACD
    df['UpperBand'], df['MiddleBand'], df['LowerBand'] = talib.BBANDS(df['close'], timeperiod=20)  			# Bollinger Bands
    df = df.dropna()  													# Drop NaN values
    return df

# replace NaN with zero in the data
def nz(value, default=0):
    if np.isnan(value):
        return default
    return value

# fetch historical data from Binance, returns a dataframe
def fetch_binance_data(symbol='BTCUSDT', interval='1d', lookback='365 days ago UTC'):
    binance_client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
    klines = binance_client.get_historical_klines(symbol, BinanceClient.KLINE_INTERVAL_1DAY, lookback)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'SMA', 'EMA', 'RSI', 'target'])
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data['close'] = data['close'].astype(float)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    binance_data = data
    return data

# fetch historical data from bitvavo and return a dataframe
def fetch_bitvavo_data(symbol='BTC-EUR', interval='1d', start_date="2024-03-15", end_date="2025-03-15"):
    bitvavo = Bitvavo({'APIKEY': BITVAVO_API_KEY,'APISECRET': BITVAVO_API_SECRET})
    params = {'market': symbol, 'interval': interval}
    if start_date:
        params['start'] = int(pd.to_datetime(start_date).timestamp() * 1000)
    if end_date:
        params['end'] = int(pd.to_datetime(end_date).timestamp() * 1000)
    response = bitvavo.candles(params['market'], params['interval'], params)
    data = pd.DataFrame(response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data['close'] = data['close'].astype(float)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    bitvavo_data = data
    return data

# Fetch historical data from yahoo finance, returning a dataframe
def fetch_yfinance_data(symbol='BTC-USD', interval='1d', period="365"):
    try:
        data= yf.download(tickers=symbol, interval=interval, period=period)
        print(f"Successfully fetched {len(data)} rows.")
        data.columns = data.columns.get_level_values(0)
        data = data.reset_index()  # Ensure 'Date' is a normal column
        data = data.rename(columns={'Date': 'timestamp'})
        data.columns = [col.lower() for col in data.columns]
        numeric_cols = ['close', 'high', 'low', 'open', 'volume']
        data['close'] = pd.to_numeric(data['close'], errors='coerce')
        data['close'] = data['close'].astype(float)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    except Exception as e:
        result = f"Error fetching data: {e}"
        return None

def fe_preprocess(exch="binance"):

  if exch=='binance':
    print('receiving Binance data, calculating indicators')
    binance_data = fetch_binance_data()
    binance_data = calculate_indicators(binance_data)
    features = ['SMA', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand'] 					# technical indicators
    scaler1 = MinMaxScaler()													# start the scaler
    binance_data[features] = scaler1.fit_transform(binance_data[features])							# apply the technical indicators to the scaler
    binance_data['target'] = binance_data['close'].shift(-1) > binance_data['close']
    binance_data['target'] = binance_data['target'].astype(int)									# force dataframe's target as type int
    binance_data['target'] = binance_data['target'].apply(lambda x: 1 if x == 1 else -1)					# Target variable (Buy=1, Hold=0, Sell=-1)
    binance_data['close'].fillna(0) 		                                                                       	 	# Fill NaN values with the last valid observation
    print(binance_data)                                                                                                  	# just for show
    return binance_data

  if exch=='bitvavo':
    print('receiving Bitvavo data, calculating indicators')
    bitvavo_data = fetch_bitvavo_data()
    bitvavo_data = calculate_indicators(bitvavo_data)
    features = ['SMA14', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand'] 				# technical indicators
    scaler2 = MinMaxScaler()													# start the scaler
    bitvavo_data[features] = scaler2.fit_transform(bitvavo_data[features])							# apply the technical indicators to the scaler
    bitvavo_data['target'] = bitvavo_data['close'].shift(-1) > bitvavo_data['close']
    bitvavo_data['target'] = bitvavo_data['target'].astype(int)									# force dataframe's target as type int
    bitvavo_data['target'] = bitvavo_data['target'].apply(lambda x: 1 if x == 1 else -1)					# Target variable (Buy=1, Hold=0, Sell=-1)
    bitvavo_data['close'].fillna(0) 		                                                                        # Fill NaN values with the last valid observation
    print(bitvavo_data)                                                                                                  	# just for show
    return bitvavo_data

  if exch=='yahoofinance':
    print('receiving Yahoo Financial data, calculating indicators')
    yf_data = fetch_yfinance_data(symbol='ETH-USD', interval="1d", period="1y")
    yf_data = calculate_indicators(yf_data)
    features = ['SMA14', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']                                 # technical indicators
    scaler3 = MinMaxScaler()                                                                                                # start the scaler
    yf_data[features] = scaler3.fit_transform(yf_data[features])
    yf_data['target'] = yf_data['close'].shift(-1) > yf_data['close']
    yf_data['target'] = yf_data['target'].astype(int)
    yf_data['target'] = yf_data['target'].apply(lambda x: 1 if x == 1 else -1)
    yf_data['close'].fillna(0)
    print(yf_data)
    return yf_data

# plotting function to display each exchange separately
# linestyles: 	linestyle='dashdot',  linestyle='dashed', linestyle='dotted', (linestyle='solid' = used for the exchange price data)
# 'SMA14' (or SMA for binance), 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand'
def plot_exchange_data(data, exchange_name, color):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.switch_backend('Qt5Agg')  # Or 'Agg', 'Qt5Agg', etc.
    # Plot price data
    ax1.plot(data['timestamp'], data['close'], label=f'{exchange_name} BTC', color=color)
    # Plot key indicators on a secondary y-axis
    ax2 = ax1.twinx()
    if exchange_name=="binance":
        ax2.plot(data['timestamp'], data['SMA'], label='SMA', linestyle='dashed', color='purple')
    else:
        ax2.plot(data['timestamp'], data['SMA14'], label='SMA14', linestyle='dashed', color='purple')

    ax2.plot(data['timestamp'], data['EMA14'], label='EMA14', linestyle='dotted', color='red')
    ax2.plot(data['timestamp'], data['MACD'], label='MACD', linestyle='dashed', color='black')
    ax2.plot(data['timestamp'], data['RSI'], label='RSI', linestyle='dashdot', color='aquamarine')
    ax2.plot(data['timestamp'], data['UpperBand'], label='UpperBand', linestyle=(0, (5, 2)), color='chocolate')
    ax2.plot(data['timestamp'], data['MiddleBand'], label='MiddleBand', linestyle=(0, (5, 10)), color='darkgoldenrod')
    ax2.plot(data['timestamp'], data['LowerBand'], label='LowerBand', linestyle=(0, (10, 5)), color='gold')

    # Labels & Legends
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Indicators')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'Historical Crypto Data: {exchange_name}')
    plt.show()

binance_data = fe_preprocess(exch='binance')
bitvavo_data = fe_preprocess(exch='bitvavo')
yf_data = fe_preprocess(exch='yahoofinance')
# Plot data separately for each exchange
plot_exchange_data(binance_data, "Binance", "blue")
plot_exchange_data(bitvavo_data, "Bitvavo", "orange")
plot_exchange_data(yf_data, "YahooFinance", "green")


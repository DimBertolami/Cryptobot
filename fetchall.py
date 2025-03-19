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

Explanation of Dependencies:
	numpy → Array operations
	pandas → Data manipulation
	matplotlib → Static plotting
	seaborn → Enhanced visualizations
	plotly → Interactive charts
	scikit-learn → ML utilities (RandomForest, train_test_split)
	xgboost → XGBoost model
	tensorflow → Deep learning models (LSTM, CNN)
'''

import os
import random
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import seaborn as sns
from ta.utils import dropna
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import requests, talib, json
import plotly.graph_objects as go
import tensorflow as tf
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from python_bitvavo_api.bitvavo import Bitvavo
from binance.client import Client as BinanceClient
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, TimeDistributed, RepeatVector, Bidirectional
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable TensorFlow oneDNN warning
warnings.filterwarnings('ignore')  # Suppress minor warnings

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device NOT found')
else:
  print('Found GPU at: {}'.format(device_name))

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY")
BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET")
ALPACA_API_KEY =  os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')

# Split Data
def split_data(df, features, target):
    X = df[features]
    y = df[target].replace({-1: 2})  # Map SELL (-1) to 2 to preserve the signal
                                     #    y = df[target].replace(-1, 0)  # Convert -1 to 0 for binary classification
    y = df[target]
    return train_test_split(X, y, test_size=0.2, shuffle=False)

'''
lr_model = train_model(LinearRegression, X_train, y_train)
LR_model = train_model(LogisticRegression, X_train, y_train)
KNC_model = train_model(KNeighborsClassifier, X_train, y_train)
DTC_model = train_model(DecisionTreeClassifier, X_train, y_train)
DTR_model = train_model(DecisionTreeRegressor, X_train, y_train)
RFR_model = train_model(RandomForestRegressor,  X_train, y_train)

'''
def train_model(model, X_train, y_train):
    if model=="LinearRegression":
        model = model()
    boolattributes = hasattr(model, 'n_estimators')
    if boolattributes==True:
        model = model(n_estimators=100)
    if boolattributes==False:
        model = model()
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    y_train = y_train.replace({-1: 0, 1: 1, 0: 2})  # Ensure labels start at 0
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', n_estimators=100, max_depth=6, max_leaves=0)
    model.fit(X_train, y_train)
    return model

# Train LSTM
def train_LSTM(X_train, y_train, lookback=365, units=50, epochs=10):
    timesteps=40           # dimensionality of the input sequence
    features=3            # dimensionality of each input representation in the sequence
    LSTMoutputDimension = 2 # dimensionality of the LSTM outputs (Hidden & Cell states)

    input = Input(shape=(timesteps, features))
    output= LSTM(LSTMoutputDimension)(input)
    model_LSTM = Model(inputs=input, outputs=output)
    W = model_LSTM.layers[1].get_weights()[0]
    U = model_LSTM.layers[1].get_weights()[1]
    b = model_LSTM.layers[1].get_weights()[2]
    print("Shapes of Matrices and Vecors:")
    print("Input [batch_size, timesteps, feature] ", input.shape)
    print("Input feature/dimension (x in formulations)", input.shape[2])
    print("Number of Hidden States/LSTM units (cells)/dimensionality of the output space (h in formulations)", LSTMoutputDimension)
    print("W", W.shape)
    print("U", U.shape)
    print("b", b.shape)
    model_LSTM.summary()
    #model = TimeDistributed([
    #    LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    #    LSTM(units),
    #    Dense(1, activation='sigmoid')
    #])
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(units),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model

# Train CNN
def train_CNN(X_train, y_train, filters=64, kernel_size=2, epochs=10):
    model = Sequential([
        Conv1D(filters, kernel_size, activation='relu', input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model

def make_decision(model, X_test):
    predictions = []

    for model in models:
        print("model: ", model)
        pred = model.predict(X_test)
        print("prediction: ", pred)
        if isinstance(pred, list):  # Ensure numpy array format
            pred = np.array(pred)
        if len(pred.shape) > 1:  # If predictions have an extra dimension, flatten them
            pred = pred.flatten()
        predictions.append(pred)

#    predictions = np.array([model.predict(X_test) for model in models])
    predictions = np.array(predictions)  # Convert to a clean numpy array
    print("predictions: ", predictions)
    final_decision = np.round(predictions.mean(axis=0))  # Ensemble averaging
    return final_decision

'''
# Strategy Execution
def make_decision(models, X_test):
    predictions = np.array([model.predict(X_test) for model in models])
    final_decision = np.round(predictions.mean(axis=0))
    return final_decision
'''
def apply_risk_management(predictions, stop_loss=0.02, take_profit=0.05):
    decisions = []
    for pred in predictions:
        pred = int(round(pred))  # Ensure integer output (no floating-point issues)
        if pred == 1:
            decisions.append("BUY")
        elif pred == -1:
            decisions.append("SELL")
        else:
            decisions.append("HOLD")
    return decisions


"""
# Risk Management
def apply_risk_management(predictions, stop_loss=0.02, take_profit=0.05):
    decisions = []
    for pred in predictions:
        if pred == 1:
            decisions.append("BUY")
        elif pred == 0:
            decisions.append("HOLD")
        else:
            decisions.append("SELL")
    return decisions
    for pred in predictions:
        if pred == 1:  # Adjusted BUY
            decisions.append("BUY")
        elif pred == 0:  # Adjusted HOLD
            decisions.append("HOLD")
        else:  # Adjusted SELL (Used to be -1, now it's 2)
            decisions.append("SELL")
    return decisions

"""
# Visualization Functions
def plot_signals(df, predictions):
    df['Decision'] = predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=df[df['Decision'] == 1].index, y=df[df['Decision'] == 1]['Close'], mode='markers', marker=dict(color='green', size=8), name='BUY'))
    fig.add_trace(go.Scatter(x=df[df['Decision'] == -1].index, y=df[df['Decision'] == -1]['Close'], mode='markers', marker=dict(color='red', size=8), name='SELL'))
    fig.update_layout(title='Trading Signals', xaxis_title='Time', yaxis_title='Price')
    fig.show()

def plot_feature_importance(model, features):
    importance = model.feature_importances_
    plt.figure(figsize=(10,5))
    sns.barplot(x=importance, y=features)
    plt.title('Feature Importance')
    plt.show()

def plot_cumulative_returns(df, predictions):
    df['Strategy Returns'] = df['Returns'] * predictions
    df['Cumulative Returns'] = (1 + df['Strategy Returns']).cumprod()
    plt.figure(figsize=(10,5))
    plt.plot(df['Cumulative Returns'], label='Strategy')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.show()

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
def fetch_binance_data(symbol='BTCUSDT', interval='1h', lookback='730 days ago UTC'):
    binance_client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
    klines = binance_client.get_historical_klines(symbol, BinanceClient.KLINE_INTERVAL_1DAY, lookback)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'SMA', 'EMA', 'RSI', 'target'])
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data['close'] = data['close'].astype(float)
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    binance_data = data
    return data

# fetch historical data from bitvavo and return a dataframe
def fetch_bitvavo_data(symbol='BTC-EUR', interval='1h', start_date="2023-03-18", end_date="2025-03-18"):
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
def fetch_yfinance_data(symbol='BTC-USD', interval='1h', period="730"):
    #try:
    data= yf.download(tickers=symbol, interval=interval, period=period)
    #print(f"Successfully fetched {len(data)} rows.")
    data.columns = data.columns.get_level_values(0)
    data = data.reset_index()  # Ensure 'Date' is a normal column
    data = data.rename(columns={'Date': 'timestamp'})
    data.columns = [col.lower() for col in data.columns]
    numeric_cols = ['close', 'high', 'low', 'open', 'volume']
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data['close'] = data['close'].astype(float)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data
    #except Exception as e:
    #    result = f"Error fetching data: {e}"
    #    return None

def fe_preprocess(exch="binance"):

  if exch=='binance':
#    print('receiving Binance data, calculating indicators')
    binance_data = fetch_binance_data()
    binance_data = calculate_indicators(binance_data)
    features = ['SMA', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand'] 					# technical indicators
    scaler = MinMaxScaler()													# start the scaler
    binance_data[features] = scaler.fit_transform(binance_data[features])							# apply the technical indicators to the scaler
    binance_data['target'] = binance_data['close'].shift(-1) > binance_data['close']
    binance_data['target'] = binance_data['target'].astype(int)									# force dataframe's target as type int
    binance_data['target'] = binance_data['target'].apply(lambda x: 1 if x == 1 else -1)					# Target variable (Buy=1, Hold=0, Sell=-1)
    binance_data['close'].fillna(0) 		                                                                       	 	# Fill NaN values with the last valid observation
    #print(binance_data)                                                                                                  	# just for show
    return binance_data

  if exch=='bitvavo':
#    print('receiving Bitvavo data, calculating indicators')
    bitvavo_data = fetch_bitvavo_data()
    bitvavo_data = calculate_indicators(bitvavo_data)
    features = ['SMA14', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand'] 				# technical indicators
    scaler = MinMaxScaler()													# start the scaler
    bitvavo_data[features] = scaler.fit_transform(bitvavo_data[features])							# apply the technical indicators to the scaler
    bitvavo_data['target'] = bitvavo_data['close'].shift(-1) > bitvavo_data['close']
    bitvavo_data['target'] = bitvavo_data['target'].astype(int)									# force dataframe's target as type int
    bitvavo_data['target'] = bitvavo_data['target'].apply(lambda x: 1 if x == 1 else -1)					# Target variable (Buy=1, Hold=0, Sell=-1)
    bitvavo_data['close'].fillna(0) 		                                                                        # Fill NaN values with the last valid observation
    #print(bitvavo_data)                                                                                                  	# just for show
    return bitvavo_data

  if exch=='yahoofinance':
#    print('receiving Yahoo Financial data, calculating indicators')
    yf_data = fetch_yfinance_data(symbol='ETH-USD', interval="1d", period="1y")
    yf_data = calculate_indicators(yf_data)
    features = ['SMA14', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']                                 # technical indicators
    scaler = MinMaxScaler()                                                                                                # start the scaler
    yf_data[features] = scaler.fit_transform(yf_data[features])
    yf_data['target'] = yf_data['close'].shift(-1) > yf_data['close']
    yf_data['target'] = yf_data['target'].astype(int)
    yf_data['target'] = yf_data['target'].apply(lambda x: 1 if x == 1 else -1)
    yf_data['close'].fillna(0)
    #print(yf_data)
    return yf_data


    # adding Visualization
    # plot_signals(df, decisions)
    # plot_feature_importance(rf_model, features)
    # plot_cumulative_returns(df, decisions)

# plotting function to display each exchange separately
# linestyles: 	linestyle='dashdot',  linestyle='dashed', linestyle='dotted', (linestyle='solid' = used for the exchange price data)
# 'SMA14' (or SMA for binance), 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand'

# Visualization Function
def plot_exchange_data(data, exchange_name, color):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Plot price data
    ax1.plot(data['timestamp'], data['close'], label=f'{exchange_name} BTC', color=color)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    # Plot key indicators on a secondary y-axis
    ax2 = ax1.twinx()
    if exchange_name == "binance":
        ax2.plot(data['timestamp'], data['SMA'], label='SMA', linestyle='dashed', color='pink')
    else:
        ax2.plot(data['timestamp'], data['SMA14'], label='SMA14', linestyle='dashed', color='pink')

    ax2.plot(data['timestamp'], data['EMA14'], label='EMA14', linestyle='dotted', color='yellow')
    ax2.plot(data['timestamp'], data['MACD'], label='MACD', linestyle='dashed', color='orange')
    ax2.plot(data['timestamp'], data['RSI'], label='RSI', linestyle='dashdot', color='aquamarine')
    ax2.plot(data['timestamp'], data['UpperBand'], label='UpperBand', linestyle=(0, (5, 2)), color='fuchsia')
    ax2.plot(data['timestamp'], data['MiddleBand'], label='MiddleBand', linestyle=(0, (5, 10)), color='darkgoldenrod')
    ax2.plot(data['timestamp'], data['LowerBand'], label='LowerBand', linestyle=(0, (10, 5)), color='gold')
    ax2.set_ylabel('Indicators')
    ax2.legend(loc='upper right')
    plt.title(f'Historical Crypto Data: {exchange_name}')
    plt.show()

binance_data = fe_preprocess(exch='binance')
features = ['SMA', 'EMA14', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']
X_train, X_test, y_train, y_test = split_data(binance_data, features, 'target')
rf_model = train_model(RandomForestClassifier, X_train, y_train)
lstm_model = train_LSTM(X_train, y_train)
cnn_model = train_CNN(X_train, y_train)
lr_model = train_model(LinearRegression, X_train, y_train)
LR_model = train_model(LogisticRegression, X_train, y_train)
KNC_model = train_model(KNeighborsClassifier, X_train, y_train)
DTC_model = train_model(DecisionTreeClassifier, X_train, y_train)
DTR_model = train_model(DecisionTreeRegressor, X_train, y_train)
RFR_model = train_model(RandomForestRegressor,  X_train, y_train)
models = [rf_model, lstm_model, cnn_model, lr_model, LR_model, KNC_model, DTC_model, DTR_model, RFR_model]
decisions = make_decision(models, X_train)
print("decisions: ", decisions)
final_trades = apply_risk_management(decisions)
print("final trade decisions: ", final_trades)


bitvavo_data = fe_preprocess(exch='bitvavo')
features = ['SMA14', 'EMA14', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']
X_train, X_test, y_train, y_test = split_data(bitvavo_data, features, 'target')

#xgb_model = train_xgboost(X_train, y_train)
rf_model = train_model(RandomForestClassifier, X_train, y_train)
lstm_model = train_LSTM(X_train, y_train)
cnn_model = train_CNN(X_train, y_train)
lr_model = train_model(LinearRegression, X_train, y_train)
LR_model = train_model(LogisticRegression, X_train, y_train)
KNC_model = train_model(KNeighborsClassifier, X_train, y_train)
DTC_model = train_model(DecisionTreeClassifier, X_train, y_train)
DTR_model = train_model(DecisionTreeRegressor, X_train, y_train)
RFR_model = train_model(RandomForestRegressor,  X_train, y_train)
models = [rf_model, lstm_model, cnn_model, lr_model, LR_model, KNC_model, DTC_model, DTR_model, RFR_model]
decisions = make_decision(models, X_train)
print("decisions: ", decisions)
final_trades = apply_risk_management(decisions)
print("final trade decisions: ", final_trades)

yf_data = fe_preprocess(exch='yahoofinance')
features = ['SMA14', 'EMA14', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']
X_train, X_test, y_train, y_test = split_data(yf_data, features, 'target')
#rf_model = train_Random_Forest(X_train, y_train)
#xgb_model = train_xgboost(X_train, y_train)
#lstm_model = train_LSTM(X_train, y_train)
#cnn_model = train_CNN(X_train, y_train)
rf_model = train_model(RandomForestClassifier, X_train, y_train)
lstm_model = train_LSTM(X_train, y_train)
cnn_model = train_CNN(X_train, y_train)
lr_model = train_model(LinearRegression, X_train, y_train)
LR_model = train_model(LogisticRegression, X_train, y_train)
KNC_model = train_model(KNeighborsClassifier, X_train, y_train)
DTC_model = train_model(DecisionTreeClassifier, X_train, y_train)
DTR_model = train_model(DecisionTreeRegressor, X_train, y_train)
RFR_model = train_model(RandomForestRegressor,  X_train, y_train)
models = [rf_model, lstm_model, cnn_model, lr_model, LR_model, KNC_model, DTC_model, DTR_model, RFR_model]
decisions = make_decision(models, X_train)
print("decisions: ", decisions)
final_trades = apply_risk_management(decisions)
print("final trade decisions: ", final_trades)
print(final_trades)

#models = [rf_model, lstm_model]
#decisions = make_decision(lstm_model, X_test)
#final_trades = apply_risk_management(decisions)
#print(final_trades)

# Plot data separately for each exchange
plot_exchange_data(binance_data, "Binance", "black")
plot_exchange_data(bitvavo_data, "Bitvavo", "black")
plot_exchange_data(yf_data, "YahooFinance", "black")


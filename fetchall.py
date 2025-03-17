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
from datetime import datetime
import matplotlib.pyplot as plt
import requests, talib, json, os
from ta.utils import dropna
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from python_bitvavo_api.bitvavo import Bitvavo
from binance.client import Client as BinanceClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY")
BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET")
ALPACA_API_KEY =  os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')

class TradingEnvironment:
    def __init__(self, data, lookback=60):
        self.data = data
        self.lookback = lookback
        self.current_step = 0
        self.balance = 1000  # Starting balance
        self.position = 0  # 0 = no position, 1 = holding asset
        self.action_space = 3  # Buy, Hold, Sell
        self.state = self.reset()

    def reset(self):
        self.current_step = self.lookback
        self.balance = 1000
        self.position = 0
        return self.data[self.current_step - self.lookback: self.current_step]

    def step(self, action):
        reward = 0
        done = False
        prev_balance = self.balance

        # Action: 0 = Buy, 1 = Hold, 2 = Sell
        current_price = self.data[self.current_step, 0]

        if action == 0:  # Buy
            if self.balance >= current_price:
                self.balance -= current_price
                self.position = 1  # Holding the asset

        elif action == 2:  # Sell
            if self.position == 1:
                self.balance += current_price
                self.position = 0

        # Reward is based on the difference in balance
        reward = self.balance - prev_balance
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            done = True

        next_state = self.data[self.current_step - self.lookback: self.current_step]
        return next_state, reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = Adam(lr=0.001)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.target_model.predict(np.reshape(next_state, [1, self.state_size]))[0])
            target_f = self.model.predict(np.reshape(state, [1, self.state_size]))
            target_f[0][action] = target
            self.model.fit(np.reshape(state, [1, self.state_size]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
#    response = bitvavo.candles({'market': symbol, 'interval': interval "1d"})
    response = bitvavo.candles(params['market'], params['interval'], params)
    data = pd.DataFrame(response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['close'] = data['close'].astype(float)
    bitvavo_data = data
    return data

# Sample DataFrame
df = pd.DataFrame({
    'close': [45.15, 46.30, 47.45, 46.85, 45.75, 44.95, 43.85, 45.00, 46.10, 46.90]
})


# create technical indicators and preprocess data 
# Features (technical indicators)  features = ['SMA14', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']
# symbol can be for binance or bitvavo (please note that they are written differently)
# Binance: symbol = 'BTCUSDT'
# Bitvavo: symbol = 'BTC-EUR'
def fe_preprocess(exch="binance"):

  if exch=='binance':
    print('receiving Binance data, calculating indicators')
    binance_data = fetch_binance_data()
    binance_data = calculate_indicators(binance_data)
    print(binance_data.head())                                                                                                  # just for show
    features = ['SMA', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand'] 					# technical indicators
    scaler1 = MinMaxScaler()													# start the scaler
    binance_data[features] = scaler1.fit_transform(binance_data[features])							# apply the technical indicators to the scaler
    binance_data['target'] = binance_data['close'].shift(-1) > binance_data['close']
    binance_data['target'] = binance_data['target'].astype(int)									# force dataframe's target as type int
    binance_data['target'] = binance_data['target'].apply(lambda x: 1 if x == 1 else -1)					# Target variable (Buy=1, Hold=0, Sell=-1)
    binance_data['close'].fillna(0) 		                                                                        # Fill NaN values with the last valid observation
    plt.plot(binance_data['timestamp'], binance_data['close'], label='Binance BTC/USDT')
    return binance_data

  if exch=='bitvavo':
    print('receiving Bitvavo data, calculating indicators')
    bitvavo_data = fetch_bitvavo_data()
    bitvavo_data = calculate_indicators(bitvavo_data)
    print(bitvavo_data.head())                                                                                                  # just for show
    features = ['SMA14', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand'] 				# technical indicators
    scaler2 = MinMaxScaler()													# start the scaler
    bitvavo_data[features] = scaler2.fit_transform(bitvavo_data[features])							# apply the technical indicators to the scaler
    bitvavo_data['target'] = bitvavo_data['close'].shift(-1) > bitvavo_data['close']
    bitvavo_data['target'] = bitvavo_data['target'].astype(int)									# force dataframe's target as type int
    bitvavo_data['target'] = bitvavo_data['target'].apply(lambda x: 1 if x == 1 else -1)					# Target variable (Buy=1, Hold=0, Sell=-1)
    bitvavo_data['close'].fillna(0) 		                                                                        # Fill NaN values with the last valid observation
    plt.plot(bitvavo_data['timestamp'], bitvavo_data['close'], label= 'Bivavo BTC/EUR')
    return bitvavo_data

##############################################################################################################################
#########################################  Here 's where the magic begins ####################################################
##############################################################################################################################
#plt.switch_backend('Agg')  # Or 'Agg', 'Qt5Agg', etc.
'''
    print(f"dataframe shape: {data.shape}")  # For DataFrame
    print(f"length: {len(data)}")   # For Series or list
    print(f"index: {data.index}")
    print(f"columns: {data.columns}")
    print(f"target data: {data['target'].iloc[i]}") 
'''

plt.figure(figsize=(12, 6))
data = fe_preprocess(exch='binance')
data = fe_preprocess(exch='bitvavo')

for i in range(364):  # Loop from 0 to 4
    for j in range(4):
        value = data['target'].iloc[j]
        if i == 364: 
            break
        if value == 1:
            print(f"{j}: buy")
            print(f"{j}:SMA          {data['SMA14']}")
            print(f"{j}:EMA14        {data['EMA14']}")
            print(f"{j}:EMA          {data['EMA']}")
            print(f"{j}:RSI          {data['RSI']}")
            print(f"{j}:MACD         {data['MACD']}")
            print(f"{j}:UpperBand    {data['UpperBand']}")
            print(f"{j}:MiddleBand   {data['MiddleBand']}")
            print(f"{j}:LowerBand    {data['LowerBand']}")
        if value == 0:
            print(f"{j}: hold")
        if value == -1:
            print(f"{j}: sell")


plt.title(' Historical Crypto Price data(By Dimi Bertolami)')
plt.xlabel('Date')
plt.ylabel('Price (usd or eur)')
plt.legend()
plt.show()

'''
    value = bitvavo_data['target'].iloc[i]
    if value == 1:
        print(f"{i} buy  {bitvavo_data['SMA']} / {binance_data['EMA']} / {binance_data['RSI']}")
    if value == 0:
        print("{i}: hold {bitvavo_data['SMA']} / {binance_data['EMA']} / {binance_data['RSI']}")
    if value == -1:
        print("{i}: sell {bitvavo_data['SMA']} / {binance_data['EMA']} / {binance_data['RSI']}")
'''

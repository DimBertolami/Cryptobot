source venvtalib/bin/activate
# python bot.py
./connect.sh
python fetchall.py

def train_Random_Forest(n_estimators=100, test_size=0.2, shuffle=False, features = ['SMA14', 'EMA14', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']):
def train_Recurrent_Neural_net_and_ltsm(feature_range=(0, 1),  lookback=60, units=50, return_sequences=True, activation='sigmoid', epochs=10, batch_size=32, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
def train_Convolutional_Neural_Net(filters=64, kernel_size=2, activation='relu',units=1, activation2='sigmoid', epochs=10, batch_size=32):
        ◦ Model: A popular approach is using Q-learning, Deep Q Networks (DQN), or Proximal Policy Optimization (PPO) to define and optimize a trading strategy. The RL agent is trained using past data to maximize long-term profit or minimize loss.
def train_dqn(df):
def train_xgboost_buy_hold_sell(df, features, target_col='target'):
def calculate_indicators(df):
def nz(value, default=0):
        return default
def fetch_binance_data(symbol='BTCUSDT', interval='1d', lookback='365 days ago UTC'):
def fetch_bitvavo_data(symbol='BTC-EUR', interval='1d', start_date="2024-03-15", end_date="2025-03-15"):
def fe_preprocess(exch="binance"):


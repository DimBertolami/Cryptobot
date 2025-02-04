import os
import requests

def fetch_alpaca_historical_data(api_key, api_secret, symbol="BTC/USD", timeframe="1Day"):
    url = f"https://data.alpaca.markets/v1beta1/crypto/BTCUSD/bars"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    params = {"symbols": symbol, "timeframe": timeframe}
    response = requests.get(url, headers=headers, params=params)
    return response.json()

# remember to store sensitive data safe inside an environment variable. 
# Do so by exporting the value from the terminal like this: 
#              export ALPACA_KEY="******************"
#              export ALPACA_SECRET="***************************"

api_key = os.getenv("ALPACA_KEY")
api_secret = os.getenv("ALPACA_SECRET")
data = fetch_alpaca_historical_data(api_key, api_secret)
print(data)

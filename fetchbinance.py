import requests
def fetch_binance_historical_data(symbol="ETHEUR", interval="1d", limit=30):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    return response.json()

data = fetch_binance_historical_data()
print(data)

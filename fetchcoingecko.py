import requests

def fetch_coingecko_historical_data(crypto_id="bitcoin", vs_currency="usd", days="30"):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    response = requests.get(url, params=params)
    return response.json()

data = fetch_coingecko_historical_data()
print(data)

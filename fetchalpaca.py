
import os
import requests
from alpaca_trade_api.rest import REST

api = REST(os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET"), base_url="https://data.alpaca.markets")
bars = api.get_crypto_bars("BTC/USD", timeframe="1Day", start="2022-01-01", end="2024-12-31").df

print(bars)
###########################################################################
# remember to store sensitive data safe inside an environment variable.   #
# Do so by exporting the value from the terminal like this:               #
#              export ALPACA_KEY="******************"                     #
#  export ALPACA_SECRET="***************************"                     #
###########################################################################

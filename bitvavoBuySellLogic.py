import os
from python_bitvavo_api import bitvavo as Bitvavo
##############################################################
# to use first:                                              #
# export BITVAVO_KEY="*****************"                     #
# export BITVAVO_SECRET="********************************"   #
##############################################################
'''
Initialization: The Bitvavo client is initialized with your API credentials.

get_balance: Retrieves the balance of a specific cryptocurrency or fiat.
get_current_price: Fetches the current market price for a trading pair (e.g., BTC-EUR).
buy_crypto and sell_crypto: Place market buy and sell orders.

    Trading Bot Logic:
    The bot continuously monitors the price.
    If the price drops below a defined threshold, it places a buy order.
    If the price rises above a defined threshold, it places a sell order.
    The thresholds can be adjusted based on your strategy (e.g., 2% in this example).

The bot runs indefinitely, checking the price every 5 seconds (adjustable via time.sleep()).


2DO:
    - Ensure proper risk management is in place. Starting with 
      a test account or small amount to minimize risk.

    - Add logging to track your trades and bot decisions.

    - Test thoroughly in simulation or with small amounts before live trading.

    - Don't annoy Bitvavo's API rate limit checker because it will block/ban me.

'''


bitvavo = {
    'APIKEY': os.getenv("BITVAVO_KEY"),
    'APISECRET': os.getenv("BITVAVO_SECRET"),
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/',
}

def get_balance(coin):
    """Fetch account balance for a specific cryptocurrency."""
    balances = bitvavo.balance({})
    for balance in balances:
        if balance['symbol'] == coin:
            return float(balance['available'])
    return 0

def get_current_price(pair):
    """Fetch the current price for a trading pair."""
    ticker = bitvavo.tickerPrice({'market': pair})
    return float(ticker['price'])

def buy_crypto(pair, amount):
    """Place a buy order."""
    try:
        response = bitvavo.placeOrder(pair, 'buy', 'market', {'amount': str(amount)})
        print("Buy Order Response:", response)
    except Exception as e:
        print("Error placing buy order:", e)

def sell_crypto(pair, amount):
    """Place a sell order."""
    try:
        response = bitvavo.placeOrder(pair, 'sell', 'market', {'amount': str(amount)})
        print("Sell Order Response:", response)
    except Exception as e:
        print("Error placing sell order:", e)

def trading_bot(pair, balance_coin, threshold=0.01):
    """
    A simple trading bot that buys when the price drops below a threshold
    and sells when it increases above the threshold.
    """
    starting_price = get_current_price(pair)
    print(f"Starting price for {pair}: {starting_price}")

    while True:
        current_price = get_current_price(pair)
        print(f"Current price: {current_price}")

        # Buy logic
        if current_price < starting_price * (1 - threshold):
            balance = get_balance(balance_coin)
            if balance > 0.001:  # Ensure there’s sufficient balance to buy
                buy_crypto(pair, balance / current_price)
                print(f"Bought {balance_coin} ( price dropped to {current_price}")
                starting_price = current_price  # Reset starting price after buying

        # Sell logic
        elif current_price > starting_price * (1 + threshold):
            crypto_balance = get_balance(pair.split('-')[0])  # Get balance of the crypto
            if crypto_balance > 0.001:  # Ensure there's sufficient balance to sell
                sell_crypto(pair, crypto_balance)
                print(f"Sold {pair.split('-')[0]} as price rose to {current_price}")
                starting_price = current_price  # Reset starting price after selling

        time.sleep(5)  # Wait for 5 seconds before checking again

# Example usage
trading_bot(pair="BTC-EUR", balance_coin="EUR", threshold=0.02)

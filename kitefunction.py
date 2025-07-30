import json
import pandas as pd
import datetime
import os
from kiteconnect import KiteConnect
from config import ACCESS_TOKEN_FILE, INSTRUMENTS_FILE

# Load instruments.csv
instruments_df = pd.read_csv(INSTRUMENTS_FILE)

def get_kite_client():
    try:
        with open(ACCESS_TOKEN_FILE, "r") as f:
            token_data = json.load(f)
        kite = KiteConnect(api_key=token_data["api_key"])
        kite.set_access_token(token_data["access_token"])
        return kite
    except Exception as e:
        print("❌ Could not load access token:", e)
        return None

def get_token_for_symbol(symbol):
    df = pd.read_csv("instruments.csv")

    row = df[df["tradingsymbol"] == symbol]
    if row.empty:
        row = df[df["name"] == symbol]

    if not row.empty:
        return int(row["instrument_token"].values[0])
    else:
        print(f"❌ Symbol not found: {symbol}")
        return None


def get_instrument_token(tradingsymbol, exchange="NFO"):
    row = instruments_df[
        (instruments_df['exchange'] == exchange) &
        (instruments_df['tradingsymbol'] == tradingsymbol)
    ]
    if not row.empty:
        return int(row.iloc[0]["instrument_token"])
    print(f"❌ Token not found for {tradingsymbol}")
    return None

def get_historical_df(instrument_token, interval, days):
    kite = get_kite_client()
    now = datetime.datetime.now()
    from_date = (now - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    to_date = now.strftime('%Y-%m-%d')
    data = kite.historical_data(instrument_token, from_date, to_date, interval)
    return pd.DataFrame(data)

def get_historical_option_price(symbol, ts):
    kite = get_kite_client()
    token = get_token_for_symbol(symbol)
    if token is None:
        return None
    from_dt = ts - datetime.timedelta(minutes=5)
    to_dt = ts
    try:
        data = kite.historical_data(token, from_dt, to_dt, "5minute")
        return data[-1]['close'] if data else None
    except Exception as e:
        print(f"⚠️ Error fetching historical price for {symbol}: {e}")
        return None

def get_quotes(symbol):
    kite = get_kite_client()
    try:
        full_symbol = f"NFO:{symbol}"
        quote = kite.ltp([full_symbol])
        return quote[full_symbol]['last_price']
    except Exception as e:
        print(f"❌ Error fetching quote for {symbol}: {e}")
        return None

def get_ltp_from_positions(tradingsymbol):
    kite = get_kite_client()
    try:
        positions = kite.positions()["net"]
        for pos in positions:
            if pos["tradingsymbol"] == tradingsymbol:
                return pos["last_price"]
    except Exception as e:
        print(f"⚠️ Error fetching LTP from positions: {e}")
    return None

def get_avgprice_from_positions(tradingsymbol):
    kite = get_kite_client()
    try:
        positions = kite.positions()["net"]
        for pos in positions:
            if pos["tradingsymbol"] == tradingsymbol:
                avg_price = pos.get("average_price", 0.0)
                qty = pos.get("quantity", 0)
                return avg_price, qty
    except Exception as e:
        print(f"⚠️ Error fetching LTP from positions: {e}")
    return None, 0

def place_option_order(tradingsymbol, qty, ordertype):
    kite = get_kite_client()
    try:
        tx_type = kite.TRANSACTION_TYPE_SELL if ordertype == "SELL" else kite.TRANSACTION_TYPE_BUY
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange="NFO",
            tradingsymbol=tradingsymbol,
            transaction_type=tx_type,
            quantity=qty,
            order_type=kite.ORDER_TYPE_MARKET,
            product=kite.PRODUCT_NRML
        )
        print(f"✅ Order Placed: {ordertype} {tradingsymbol} | Order ID: {order_id}")
        return order_id
    except Exception as e:
        print(f"❌ Order placement failed for {tradingsymbol}: {e}")
        return None




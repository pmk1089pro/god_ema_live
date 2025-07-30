# emalive.py

import os
import time
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import sqlite3
import logging
from config import SYMBOL, INTERVAL,NEAREST_LTP, CANDLE_DAYS as DAYS, REQUIRED_CANDLES, QTY, DB_FILE, LOG_FILE
from kitefunction import get_historical_df, get_ltp_from_positions, place_option_order, get_token_for_symbol, get_kite_client,get_quotes,get_avgprice_from_positions
from telegrambot import send_telegram_message

# ====== Setup Logging ======
logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

instrument_token = get_token_for_symbol(SYMBOL)

# ====== DB Setup ======
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal TEXT,
            spot_entry REAL,
            option_symbol TEXT,
            strike INTEGER,
            expiry TEXT,
            option_sell_price REAL,
            entry_time TEXT,
            spot_exit REAL,
            option_buy_price REAL,
            exit_time TEXT,
            pnl REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS open_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal TEXT,
            spot_entry REAL,
            option_symbol TEXT,
            strike INTEGER,
            expiry TEXT,
            option_sell_price REAL,
            entry_time TEXT
        )
    """)
    conn.commit()
    conn.close()

# ====== EMA Signal Logic ======


def generate_signals(df, len1=8, len2=20):
    # === Calculate EMAs ===
    df['ema1'] = df['close'].ewm(span=len1, adjust=False).mean()
    df['ema2'] = df['close'].ewm(span=len2, adjust=False).mean()

    # === Detect crossovers ===
    df['crossover'] = (df['ema1'] > df['ema2']) & (df['ema1'].shift(1) <= df['ema2'].shift(1))
    df['crossunder'] = (df['ema1'] < df['ema2']) & (df['ema1'].shift(1) >= df['ema2'].shift(1))

    # === Track trend like Pine Script's var ===
    trend = []
    current_trend = 0
    for i in range(len(df)):
        if df['crossover'].iloc[i]:
            current_trend = 1
        elif df['crossunder'].iloc[i]:
            current_trend = -1
        trend.append(current_trend)
    df['trend'] = trend

    # === Two-candle confirmation logic ===
    df['twoAbove'] = (df['close'] > df['ema1']) & (df['close'] > df['ema2']) & \
                     (df['close'].shift(1) > df['ema1'].shift(1)) & (df['close'].shift(1) > df['ema2'].shift(1))

    df['twoBelow'] = (df['close'] < df['ema1']) & (df['close'] < df['ema2']) & \
                     (df['close'].shift(1) < df['ema1'].shift(1)) & (df['close'].shift(1) < df['ema2'].shift(1))

    # === Signal detection with one signal per trend direction ===
    buy_signals = []
    sell_signals = []
    buy_fired = False
    sell_fired = False
    prev_trend = 0

    for i in range(len(df)):
        curr_trend = df['trend'][i]
        new_cross = curr_trend != prev_trend

        if new_cross:
            buy_fired = False
            sell_fired = False

        buy = curr_trend == 1 and df['twoAbove'][i] and not buy_fired
        sell = curr_trend == -1 and df['twoBelow'][i] and not sell_fired

        buy_signals.append(buy)
        sell_signals.append(sell)

        if buy:
            buy_fired = True
        if sell:
            sell_fired = True

        prev_trend = curr_trend

    df['buySignal'] = buy_signals
    df['sellSignal'] = sell_signals

    return df
# ====== Utilities ======
def wait_until_next_candle():
    now = datetime.datetime.now()
    gap = int(INTERVAL.replace("minute", ""))
    wait_sec = (gap - (now.minute % gap)) * 60 - now.second + 2
    time.sleep(max(2, wait_sec))

def get_option_symbol(signal, spot):
    df = pd.read_csv("instruments.csv")
    
    # Calculate strike based on signal
    strike = int(round(spot / 100.0) * 100) + (-600 if signal == "BUY" else 600)
    opt_type = "PE" if signal == "BUY" else "CE"

    # Filter for relevant option contracts
    df = df[
        (df['name'] == SYMBOL) &
        (df['segment'] == 'NFO-OPT') &
        (df['strike'] == strike) &
        (df['tradingsymbol'].str.endswith(opt_type))
    ].copy()

    if df.empty:
        print(f"⚠️ No options found for strike {strike}{opt_type}")
        return None, None, None, None

    # Parse expiry
    df['expiry'] = pd.to_datetime(df['expiry'])

    # Get today's date and calculate next-to-next Thursday
    today = pd.Timestamp.today().normalize()
    days_until_thursday = (3 - today.weekday() + 7) % 7  # 3 = Thursday
    this_week_thursday = today + timedelta(days=days_until_thursday)
    target_expiry = this_week_thursday + timedelta(days=14)

    # Get the week range (Monday to Sunday) for the expiry week
    week_start = target_expiry - timedelta(days=target_expiry.weekday())
    week_end = week_start + timedelta(days=6)

    # Filter options within that week
    df_target = df[(df['expiry'] >= week_start) & (df['expiry'] <= week_end)].sort_values('expiry')

    if df_target.empty:
        print(f"❌ No options found between {week_start.date()} and {week_end.date()} for strike {strike}{opt_type}")
        return None, None, None, None

    # Pick the earliest expiry in that week
    option = df_target.iloc[0]
    ltp = get_quotes(option['tradingsymbol']) or 0.0

    return option['tradingsymbol'], strike, option['expiry'].strftime('%Y-%m-%d'), ltp

def get_optimal_option(signal, spot, nearest_price):
    strike = int(round(spot / 100.0) * 100)
    print(f"Signal: {signal}, Spot: {spot}, Nearest 100 Strike: {strike}")

    df = pd.read_csv("instruments.csv")

    best_option = None
    best_ltp_diff = float('inf')

    while True:
        # Determine option type and adjust strike
        if signal == "BUY":
            opt_type = "PE"
            strike -= 100
        else:
            opt_type = "CE"
            strike += 100

        # Filter instruments for symbol, segment, strike, and option type
        df_filtered = df[
            (df['name'] == 'NIFTY') &
            (df['segment'] == 'NFO-OPT') &
            (df['strike'] == strike) &
            (df['tradingsymbol'].str.endswith(opt_type))
        ].copy()

        if df_filtered.empty:
            print(f"⚠️ No options found for strike {strike}{opt_type}")
            break

        # Parse expiry as datetime
        df_filtered['expiry'] = pd.to_datetime(df_filtered['expiry'])
        today = pd.Timestamp.today().normalize()

        # === ✅ Calculate target expiry week (next-to-next Thursday) ===
        days_until_thursday = (3 - today.weekday() + 7) % 7  # 3 = Thursday
        this_week_thursday = today + timedelta(days=days_until_thursday)
        target_expiry = this_week_thursday + timedelta(days=14)

        # Get Monday–Sunday for that target expiry week
        week_start = target_expiry - timedelta(days=target_expiry.weekday())  # Monday
        week_end = week_start + timedelta(days=6)  # Sunday

        # Filter options within the expiry week
        target_options = df_filtered[
            (df_filtered['expiry'] >= week_start) &
            (df_filtered['expiry'] <= week_end)
        ].sort_values('expiry')

        if target_options.empty:
            print(f"❌ No options found in week {week_start.date()} to {week_end.date()} for strike {strike}{opt_type}")
            break

        # Pick the earliest expiry in that week (may be Wednesday if Thursday is holiday)
        opt = target_options.iloc[0]
        opt_symbol = opt['tradingsymbol']
        expiry = opt['expiry'].strftime('%Y-%m-%d')

        # Get live LTP for the option
        ltp = get_quotes(opt_symbol) or 0.0

        # Track the option closest to nearest_price
        diff = abs(ltp - nearest_price)
        if diff < best_ltp_diff:
            best_ltp_diff = diff
            best_option = (opt_symbol, strike, expiry, ltp)
        else:
            break  # Stop when LTP moves away from nearest_price

    if best_option:
        return best_option
    else:
        return None, None, None, None

def save_open_position(trade):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO open_positions (signal, spot_entry, option_symbol, strike, expiry, option_sell_price, entry_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        trade["Signal"], trade["SpotEntry"], trade["OptionSymbol"], trade["Strike"],
        trade["Expiry"], trade["OptionSellPrice"], trade["EntryTime"]
    ))
    conn.commit()
    conn.close()

def delete_open_position(symbol):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM open_positions WHERE option_symbol = ?", (symbol,))
    conn.commit()
    conn.close()

def load_open_position():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT signal, spot_entry, option_symbol, strike, expiry, option_sell_price, entry_time FROM open_positions ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "Signal": row[0],
            "SpotEntry": row[1],
            "OptionSymbol": row[2],
            "Strike": row[3],
            "Expiry": row[4],
            "OptionSellPrice": row[5],
            "EntryTime": row[6]
        }
    return None

def record_trade(trade):
    print(f"✅ Recording trade: {trade}")
    logging.info(f"✅ Recording trade: {trade}")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO live_trades (signal, spot_entry, option_symbol, strike, expiry, option_sell_price,
        entry_time, spot_exit, option_buy_price, exit_time, pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade["Signal"], trade["SpotEntry"], trade["OptionSymbol"], trade["Strike"],
        trade["Expiry"], trade["OptionSellPrice"], trade["EntryTime"],
        trade["SpotExit"], trade["OptionBuyPrice"], trade["ExitTime"], trade["PnL"]
    ))
    conn.commit()
    conn.close()
    print(f"📊 Trade recorded in DB.")


# ====== Market Hours ======
def is_market_open():
    now = datetime.datetime.now().time()
    return datetime.time(9, 15) <= now <= datetime.time(15, 30)
    # return True  # For testing purposes, assume market is always open

# ====== Main Live Trading Loop ======
def live_trading():
    # kite = get_kite_client()
    open_trade = load_open_position()
    
    if open_trade:
        trade = open_trade
        position = open_trade["Signal"]
        logging.info(f"📌 Resumed open position: {position} | {open_trade['OptionSymbol']} @ ₹{open_trade['OptionSellPrice']}")
        print(f"➡️ Loaded open position: {open_trade}")
        send_telegram_message(f"📌 PMK 5 Resumed open position: {position} | {open_trade['OptionSymbol']} @ ₹{open_trade['OptionSellPrice']}")
    else:
        trade = {}
        position = None
        print("ℹ️ No open position. Waiting for next signal...")
        logging.info("ℹ️ No open position. Waiting for next signal...")

    # Show market status
    if not is_market_open():
        print("🕒 PMK 5 Market is currently closed. Live trading will start once the market opens.")
        send_telegram_message("🕒 PMK 5 Market is currently closed. Live trading will start once the market opens.")

    while True:
        try:
            if not is_market_open():
                print("⏳PMK 5  Waiting for market to open...")
                time.sleep(60)
                continue

            # ----- Fetch Historical Data -----
            df = get_historical_df(instrument_token, INTERVAL, DAYS)
            print(f"🕵️‍♀️ PMK 5 Candles available: {len(df)} / Required: {REQUIRED_CANDLES}")

            if len(df) < REQUIRED_CANDLES:
                print("⚠️ PMK 5 Not enough candles. Waiting...")
                time.sleep(60)
                continue

            # ----- Generate Signal -----
            df = generate_signals(df)
            latest = df.iloc[-1]
            ts = latest['date'].strftime('%Y-%m-%d %H:%M')
            close = latest['close']

            logging.info(f"{ts} | Close: {close} | Buy: {latest['buySignal']} | Sell: {latest['sellSignal']}")
            print(f"{ts} | Close: {close} | Buy: {latest['buySignal']} | Sell: {latest['sellSignal']}")

            # ----- Trade Logic -----
            # ✅ Exit previous position
            if latest['buySignal'] and position != "BUY":
                if position == "SELL":
                    trade.update({
                        "SpotExit": close,
                        "ExitTime": ts,
                        "OptionBuyPrice": get_quotes(trade["OptionSymbol"]),
                    })
                    trade["PnL"] = trade["OptionSellPrice"] - trade["OptionBuyPrice"]
                    # place_option_order(trade["OptionSymbol"], QTY, "BUY")
                    record_trade(trade)
                    delete_open_position(trade["OptionSymbol"])
                    send_telegram_message(f"📤 PMK 5 Exit SELL\n{trade['OptionSymbol']} @ ₹{trade['OptionBuyPrice']:.2f}")

                # Enter new BUY position
                opt_symbol, strike, expiry, ltp = get_optimal_option("BUY", close, NEAREST_LTP)
                # opt_symbol, strike, expiry, ltp = get_option_symbol("BUY", close)

                option_price = ltp
                # place_option_order(opt_symbol, QTY, "SELL")

                trade = {
                    "Signal": "BUY", "SpotEntry": close, "OptionSymbol": opt_symbol,
                    "Strike": strike, "Expiry": expiry, "OptionSellPrice": option_price, "EntryTime": ts
                }
                save_open_position(trade)
                position = "BUY"
                send_telegram_message(f"🟢 PMK 5 Buy Signal\n{opt_symbol} | LTP ₹{option_price:.2f}")

            elif latest['sellSignal'] and position != "SELL":
                if position == "BUY":
                    trade.update({
                        "SpotExit": close,
                        "ExitTime": ts,
                        "OptionBuyPrice": get_quotes(trade["OptionSymbol"]),
                    })
                    trade["PnL"] = trade["OptionSellPrice"] - trade["OptionBuyPrice"]
                    # place_option_order(trade["OptionSymbol"], QTY, "BUY")
                    
                    record_trade(trade)
                    delete_open_position(trade["OptionSymbol"])
                    send_telegram_message(f"📤 PMK 5 Exit BUY\n{trade['OptionSymbol']} @ ₹{trade['OptionBuyPrice']:.2f}")

                # Enter new SELL position
                opt_symbol, strike, expiry, ltp = get_optimal_option("SELL", close, NEAREST_LTP)
                # opt_symbol, strike, expiry, ltp = get_option_symbol("SELL", close)

                option_price = ltp
                # place_option_order(opt_symbol, QTY, "SELL")
                trade = {
                    "Signal": "SELL", "SpotEntry": close, "OptionSymbol": opt_symbol,
                    "Strike": strike, "Expiry": expiry, "OptionSellPrice": option_price, "EntryTime": ts
                }
                save_open_position(trade)
                position = "SELL"
                send_telegram_message(f"🔴 PMK 5 Sell Signal\n{opt_symbol} | LTP ₹{option_price:.2f}")

            wait_until_next_candle()

        except Exception as e:
            logging.error(f"Exception: {e}", exc_info=True)
            send_telegram_message(f"⚠️ PMK 5 Error: {e}")
            time.sleep(60)

# ====== Run ======
if __name__ == "__main__":
    init_db()
    send_telegram_message("🚀 PMK 5 Live trading started!")
    # SPOT = 24719
    # opt_sym = get_optimal_option("BUY", SPOT,40) 
    # print(opt_sym)
    # print(get_option_symbol("BUY", SPOT))
    # avg_price,avg_qty = get_avgprice_from_positions("NIFTY25AUG23700PE")
    # print(f"Avg Price: {avg_price}, Qty: {avg_qty}")
    live_trading()

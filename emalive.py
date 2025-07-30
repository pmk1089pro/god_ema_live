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
            pnl REAL,
            qty REAL
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
            entry_time TEXT,
            qty REAL
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
        INSERT INTO open_positions (signal, spot_entry, option_symbol, strike, expiry, option_sell_price, entry_time,qty)
        VALUES (?, ?, ?, ?, ?, ?, ?,?)
    """, (
        trade["Signal"], trade["SpotEntry"], trade["OptionSymbol"], trade["Strike"],
        trade["Expiry"], trade["OptionSellPrice"], trade["EntryTime"],trade["qty"]
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
    c.execute("SELECT signal, spot_entry, option_symbol, strike, expiry, option_sell_price, entry_time,qty FROM open_positions ORDER BY id DESC LIMIT 1")
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
            "EntryTime": row[6],
            "qty": row[7] 
        }
    return None

def record_trade(trade):
    print(f"✅ Recording trade: {trade}")
    logging.info(f"✅ Recording trade: {trade}")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO live_trades (signal, spot_entry, option_symbol, strike, expiry, option_sell_price,
        entry_time, spot_exit, option_buy_price, exit_time, pnl,qty)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
    """, (
        trade["Signal"], trade["SpotEntry"], trade["OptionSymbol"], trade["Strike"],
        trade["Expiry"], trade["OptionSellPrice"], trade["EntryTime"],
        trade["SpotExit"], trade["OptionBuyPrice"], trade["ExitTime"], trade["PnL"],trade["qty"]
    ))
    conn.commit()
    conn.close()
    print(f"📊 Trade recorded in DB.")

def get_next_expiry_optimal_option(signal, last_expiry, price, nearest_price):
    """
    Get the next expiry option after last_expiry with nearest strike to price
    and closest LTP to nearest_price.
    
    Args:
        signal (str): "BUY" or "SELL"
        last_expiry (str): Last expiry date in 'YYYY-MM-DD' format
        price (float): Current spot price
        nearest_price (float): Target LTP to match
    
    Returns:
        tuple: (option_symbol, strike, expiry, ltp) or (None, None, None, None) if not found
    """
    try:
        df = pd.read_csv("instruments.csv")
        
        # Calculate base strike (nearest hundred)
        base_strike = int(round(price / 100.0) * 100)
        
        # Determine option type based on signal
        opt_type = "PE" if signal == "BUY" else "CE"
        
        # Parse last expiry date
        last_expiry_date = pd.to_datetime(last_expiry)
        
        # Initialize strike adjustment
        if signal == "BUY":
            strike = base_strike - 100
            strike_adjustment = -100
        else:  # SELL
            strike = base_strike + 100
            strike_adjustment = 100
        
        best_option = None
        best_ltp_diff = float('inf')
        previous_strike = None
        previous_ltp = None
        
        while True:
            #print(f"🔍 Checking strike {strike}{opt_type} for signal {signal}")
            
            # Filter for relevant option contracts
            df_filtered = df[
                (df['name'] == SYMBOL) &
                (df['segment'] == 'NFO-OPT') &
                (df['strike'] == strike) &
                (df['tradingsymbol'].str.endswith(opt_type))
            ].copy()
            
            if df_filtered.empty:
                #print(f"⚠️ No options found for strike {strike}{opt_type}")
                break
            
            # Parse expiry dates
            df_filtered['expiry'] = pd.to_datetime(df_filtered['expiry'])
            
            # Filter for expiries after the last expiry
            df_next_expiry = df_filtered[df_filtered['expiry'] > last_expiry_date].copy()
            
            if df_next_expiry.empty:
                #print(f"❌ No expiries found after {last_expiry} for strike {strike}{opt_type}")
                break
            
            # Sort by expiry date to get the next available expiry
            df_next_expiry = df_next_expiry.sort_values('expiry')
            
            # Get the next expiry date
            next_expiry = df_next_expiry.iloc[0]['expiry']
            
            # Filter options for this specific expiry
            df_same_expiry = df_next_expiry[df_next_expiry['expiry'] == next_expiry].copy()
            
            if df_same_expiry.empty:
                #print(f"❌ No options found for expiry {next_expiry} at strike {strike}{opt_type}")
                break
            
            # Get LTP for this strike
            option = df_same_expiry.iloc[0]
            opt_symbol = option['tradingsymbol']
            ltp = get_quotes(opt_symbol) or 0.0
            
            #print(f"📊 Strike {strike}{opt_type}: LTP = ₹{ltp:.2f}")
            
            # Check if LTP is greater than 40
            if ltp > 40:
                #print(f"⚠️ LTP ₹{ltp:.2f} > 40, adjusting strike by {strike_adjustment}")
                previous_strike = strike
                previous_ltp = ltp
                previous_symbol = opt_symbol
                strike += strike_adjustment
                continue
            
            # LTP is <= 40, now compare with previous strike if available
            if previous_strike is not None and previous_ltp is not None:
                #print(f"🔄 Comparing strikes: {previous_strike}(₹{previous_ltp:.2f}) vs {strike}(₹{ltp:.2f})")
                
                # Calculate which strike is closer to nearest_price
                prev_diff = abs(previous_ltp - nearest_price)
                curr_diff = abs(ltp - nearest_price)
                
                if prev_diff <= curr_diff:
                    # Previous strike is better
                    best_strike = previous_strike
                    best_ltp = previous_ltp
                    best_symbol = previous_symbol
                    #print(f"✅ Selected previous strike {previous_strike} with LTP ₹{previous_ltp:.2f}")
                else:
                    # Current strike is better
                    best_strike = strike
                    best_ltp = ltp
                    best_symbol = opt_symbol
                    #print(f"✅ Selected current strike {strike} with LTP ₹{ltp:.2f}")
            else:
                # No previous strike to compare, use current
                best_strike = strike
                best_ltp = ltp
                best_symbol = opt_symbol
                #print(f"✅ Selected strike {strike} with LTP ₹{ltp:.2f}")
            
            # Get the expiry date for the selected option
            
            best_expiry = next_expiry.strftime('%Y-%m-%d')
            
            return best_symbol, best_strike, best_expiry, best_ltp
        
        # If we reach here, no suitable option found
        print(f"❌ No suitable option found for signal {signal} after {last_expiry}")
        return None, None, None, None
            
    except Exception as e:
        print(f"❌ Error in get_next_expiry_optimal_option: {e}")
        logging.error(f"Error in get_next_expiry_optimal_option: {e}")
        return None, None, None, None

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
        logging.info(f"📌 PMK {INTERVAL} Resumed open position: {position} | {open_trade['OptionSymbol']} @ ₹{open_trade['OptionSellPrice']}| Qty: {open_trade['qty']}")
        print(f"➡️ PMK {INTERVAL} Loaded open position: {open_trade}")
        send_telegram_message(f"📌 PMK {INTERVAL} Resumed open position: {position} | {open_trade['OptionSymbol']} @ ₹{open_trade['OptionSellPrice']}| Qty: {open_trade['qty']}")
    else:
        trade = {}
        position = None
        print("ℹ️ PMK 5 No open position. Waiting for next signal...")
        logging.info("ℹ️ PMK 5 No open position. Waiting for next signal...")

    if not is_market_open():
        print(f"🕒 PMK  {INTERVAL} Market is currently closed. Live trading will start once the market opens.")
        send_telegram_message(f"🕒 PMK {INTERVAL} Market is currently closed. Live trading will start once the market opens.")

    while True:
        try:
            if not is_market_open():
                print(f"PMK {INTERVAL}  Waiting for market to open...")
                time.sleep(60)
                continue

            df = get_historical_df(instrument_token, INTERVAL, DAYS)
            print(f"🕵️‍♀️ PMK {INTERVAL}  Candles available: {len(df)} / Required: {REQUIRED_CANDLES}")

            if len(df) < REQUIRED_CANDLES:
                print(f"⚠️ PMK {INTERVAL} Not enough candles. Waiting...")
                time.sleep(60)
                continue

            df = generate_signals(df)
            latest = df.iloc[-1]
            ts = latest['date'].strftime('%Y-%m-%d %H:%M')
            close = latest['close']

            logging.info(f"{ts} | Close: {close} | Buy: {latest['buySignal']} | Sell: {latest['sellSignal']}")
            print(f"{ts} | Close: {close} | Buy: {latest['buySignal']} | Sell: {latest['sellSignal']}")

            # ✅ Exit and Enter BUY
            if latest['buySignal'] and position != "BUY":
                if position == "SELL":
                    trade.update({
                        "SpotExit": close,
                        "ExitTime": ts,
                        "OptionBuyPrice": get_quotes(trade["OptionSymbol"]),
                    })
                    trade["PnL"] = trade["OptionSellPrice"] - trade["OptionBuyPrice"]
                    # place_option_order(trade["OptionSymbol"], trade["qty"], "BUY")
                    record_trade(trade)
                    delete_open_position(trade["OptionSymbol"])
                    send_telegram_message(f"📤 PMK {INTERVAL} Exit SELL\n{trade['OptionSymbol']} @ ₹{trade['OptionBuyPrice']:.2f}")

                # opt_symbol, strike, expiry, ltp = get_optimal_option("BUY", close, NEAREST_LTP)
                result = get_optimal_option("BUY", close, NEAREST_LTP)
                if result is None or result[0] is None:
                    # Handle: No suitable option
                    logging.error(f"❌ PMK {INTERVAL}: No suitable option found for BUY signal.")
                    send_telegram_message(f"❌ PMK {INTERVAL}: No suitable option found for BUY signal.")
                    # Maybe continue or break
                    continue
                else:
                    opt_symbol, strike, expiry, ltp = result
                # place_option_order(opt_symbol, qty, "SELL")
                time.sleep(2)  # Allow order to execute

                avg_price, qty = get_avgprice_from_positions(opt_symbol)
                if avg_price is None:
                    avg_price = ltp
                    qty = QTY

                trade = {
                    "Signal": "BUY", "SpotEntry": close, "OptionSymbol": opt_symbol,
                    "Strike": strike, "Expiry": expiry,
                    "OptionSellPrice": avg_price, "EntryTime": ts,
                    "qty": qty
                }
                save_open_position(trade)
                position = "BUY"
                send_telegram_message(f"🟢 PMK {INTERVAL} Buy Signal\n{opt_symbol} | Avg ₹{avg_price:.2f} | Qty: {qty}")

            # ✅ Exit and Enter SELL
            elif latest['sellSignal'] and position != "SELL":
                if position == "BUY":
                    trade.update({
                        "SpotExit": close,
                        "ExitTime": ts,
                        "OptionBuyPrice": get_quotes(trade["OptionSymbol"]),
                    })
                    trade["PnL"] = trade["OptionSellPrice"] - trade["OptionBuyPrice"]
                    trade["qty"] = trade.get("qty", QTY)

                    # place_option_order(trade["OptionSymbol"], trade["qty"], "BUY")
                    record_trade(trade)
                    delete_open_position(trade["OptionSymbol"])
                    send_telegram_message(f"📤 PMK {INTERVAL} Exit BUY\n{trade['OptionSymbol']} @ ₹{trade['OptionBuyPrice']:.2f}")

                # opt_symbol, strike, expiry, ltp = get_optimal_option("SELL", close, NEAREST_LTP)
                    result = get_optimal_option("SELL", close, NEAREST_LTP)
                    if result is None or result[0] is None:
                        # Handle: No suitable option
                        logging.error(f"❌ PMK {INTERVAL}: No suitable option found for SELL signal.")
                        send_telegram_message(f"❌ PMK {INTERVAL}: No suitable option found for SELL signal.")
                        # Maybe continue or break
                        continue
                    else:
                        opt_symbol, strike, expiry, ltp = result
                    # place_option_order(opt_symbol, QTY, "SELL")
                    time.sleep(2)

                avg_price, qty = get_avgprice_from_positions(opt_symbol)
                if avg_price is None:
                    avg_price = ltp
                    qty = QTY

                trade = {
                    "Signal": "SELL", "SpotEntry": close, "OptionSymbol": opt_symbol,
                    "Strike": strike, "Expiry": expiry,
                    "OptionSellPrice": avg_price, "EntryTime": ts,
                    "qty": qty
                }
                save_open_position(trade)
                position = "SELL"
                send_telegram_message(f"🔴 PMK {INTERVAL} Sell Signal\n{opt_symbol} | Avg ₹{avg_price:.2f} | Qty: {qty}")

            # ✅ Monitor Open Position for Target Exit
            elif trade and "OptionSymbol" in trade and "OptionSellPrice" in trade:
                current_ltp = get_quotes(trade["OptionSymbol"])
                entry_ltp = trade["OptionSellPrice"]

                if entry_ltp != 0.0 and current_ltp <= 0.6 * entry_ltp:
                    trade["SpotExit"] = close
                    trade["ExitTime"] = ts
                    trade["OptionBuyPrice"] = current_ltp
                    trade["PnL"] = entry_ltp - current_ltp
                    record_trade(trade)
                    delete_open_position(trade["OptionSymbol"])
                    send_telegram_message(f"📤 PMK {INTERVAL} Exit {trade['Signal']}\n{trade['OptionSymbol']} @ ₹{current_ltp:.2f}")
                    logging.info(f"🔴 PMK {INTERVAL} Target triggered for {trade['OptionSymbol']} at ₹{current_ltp:.2f}")
                    last_expiry = trade["Expiry"]
                    signal = trade["Signal"]
                    trade = {}

                    # opt_symbol, strike, expiry, ltp = get_next_expiry_optimal_option(signal, last_expiry, close, NEAREST_LTP)
                    result = get_next_expiry_optimal_option(signal, last_expiry, close, NEAREST_LTP)
                    if result is None or result[0] is None:
                        logging.error(f"❌ No expiry found after {last_expiry} for reentry.")
                        position = None
                        continue
                    else:
                        opt_symbol, strike, expiry, ltp = result
                    # place_option_order(opt_symbol, QTY, "SELL")
                    time.sleep(2)

                    avg_price, qty = get_avgprice_from_positions(opt_symbol)
                    if avg_price is None:
                        avg_price = ltp
                        qty = QTY

                    if opt_symbol:
                        trade = {
                            "Signal": signal,
                            "SpotEntry": close,
                            "OptionSymbol": opt_symbol,
                            "Strike": strike,
                            "Expiry": expiry,
                            "OptionSellPrice": avg_price,
                            "EntryTime": ts,
                            "qty": qty
                        }
                        save_open_position(trade)
                        send_telegram_message(f"🔁 PMK {INTERVAL} Reentry {signal}\n{opt_symbol} | Avg ₹{avg_price:.2f} | Qty: {qty}")
                        position = signal
                    else:
                        logging.info(f"❌ No expiry found after {last_expiry} for reentry.")
                        position = None

            wait_until_next_candle()

        except Exception as e:
            logging.error(f"Exception: {e}", exc_info=True)
            send_telegram_message(f"⚠️ PMK {INTERVAL} Error: {e}")
            time.sleep(60)


# ====== Run ======
if __name__ == "__main__":
    init_db()
    send_telegram_message(f"🚀 PMK {INTERVAL} Live trading started!")
    live_trading()
    # SPOT = 24870
    # print(get_optimal_option("BUY", SPOT, NEAREST_LTP))
    # print(get_avgprice_from_positions("NIFTY2581424200PE"))
    # avg_price, qty = get_avgprice_from_positions("NIFTY2581424200PE")
    # print (avg_price, qty)

import yfinance as yf
import pandas as pd
import numpy as np

def analyze_ihsg_behavior(ticker_input):
    # 1. Auto-Format for IHSG
    ticker = ticker_input.upper()
    if not ticker.endswith(".JK"):
        ticker += ".JK"
    
    print(f"\n--- 🇮🇩 Analyzing {ticker} (Last 2 Years) ---")
    
    # 2. Fetch Data
    try:
        df = yf.download(ticker, period="2y", progress=False)
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            print(f"❌ Error: Data not found. Is '{ticker}' delisted or suspended?")
            return
            
        # --- IHSG SPECIFIC: LIQUIDITY CHECK ---
        avg_volume_value = (df['Close'] * df['Volume']).tail(20).mean()
        if avg_volume_value < 5_000_000_000: # 5 Billion IDR daily avg
            print(f"⚠️ WARNING: This stock has low liquidity (Avg Daily Value < 5M IDR).")
            print("   Technical signals (MA/RSI) may be unreliable due to 'Bandar' manipulation.")
            print("   Proceed with skepticism.\n")
            
    except Exception as e:
        print(f"Error: {e}")
        return

    # 3. Calculate Indicators
    df['MA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['MA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['MA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = df['EMA12'] - df['EMA26']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4. Define Signals & Retest (Look forward 5 days)
    df['Prev_MA9'] = df['MA9'].shift(1)
    df['Prev_MA21'] = df['MA21'].shift(1)
    df['Prev_MA50'] = df['MA50'].shift(1)
    df['Prev_MACD'] = df['MACD_Line'].shift(1)
    df['Prev_RSI'] = df['RSI'].shift(1)

    signals = []
    look_forward = 5  # Check result 5 days later

    for i in range(50, len(df) - look_forward):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Performance Check
        future_price = df.iloc[i + look_forward]['Close']
        entry_price = curr['Close']
        pct_change = ((future_price - entry_price) / entry_price) * 100
        
        # --- SIGNAL LOGIC ---
        
        # Signal 1: MA 9/21 Cross
        if prev['Prev_MA9'] < prev['Prev_MA21'] and curr['MA9'] > curr['MA21']:
            signals.append({'Signal': 'MA 9 Cross Over 21 (Bull)', 'Return_5d': pct_change})
        elif prev['Prev_MA9'] > prev['Prev_MA21'] and curr['MA9'] < curr['MA21']:
            signals.append({'Signal': 'MA 9 Cross Under 21 (Bear)', 'Return_5d': pct_change})

        # Signal 2: MA 21/50 Cross (Trend)
        if prev['Prev_MA21'] < prev['Prev_MA50'] and curr['MA21'] > curr['MA50']:
            signals.append({'Signal': 'MA 21 Cross Over 50 (Bull)', 'Return_5d': pct_change})
        
        # Signal 3: MACD Zero Line
        if prev['Prev_MACD'] < 0 and curr['MACD_Line'] > 0:
            signals.append({'Signal': 'MACD Zero Cross Up (Bull)', 'Return_5d': pct_change})
        elif prev['Prev_MACD'] > 0 and curr['MACD_Line'] < 0:
            signals.append({'Signal': 'MACD Zero Cross Down (Bear)', 'Return_5d': pct_change})

        # Signal 4: RSI Extremes
        if prev['Prev_RSI'] < 70 and curr['RSI'] >= 70:
            signals.append({'Signal': 'RSI Enter Overbought (>70)', 'Return_5d': pct_change})
        elif prev['Prev_RSI'] > 30 and curr['RSI'] <= 30:
            signals.append({'Signal': 'RSI Enter Oversold (<30)', 'Return_5d': pct_change})

    # 5. Summarize Results
    results_df = pd.DataFrame(signals)
    
    if results_df.empty:
        print("No significant signals found in the last 2 years.")
        return

    # Advanced Grouping
    summary = results_df.groupby('Signal')['Return_5d'].agg(
        Count='count',
        Avg_Return_5d='mean',
        Win_Rate=lambda x: (x > 0).mean() * 100
    ).sort_values(by='Count', ascending=False)
    
    # Formatting for readability
    pd.options.display.float_format = '{:.2f}'.format
    print(f"\n=== BEHAVIOR REPORT FOR {ticker} ===")
    print(summary)
    
    # Specific Advice Generator
    print(f"\n--- 🧠 AI Skeptic's Analysis for {ticker} ---")
    
    # Check RSI Behavior
    rsi_data = results_df[results_df['Signal'] == 'RSI Enter Overbought (>70)']
    if not rsi_data.empty:
        rsi_avg = rsi_data['Return_5d'].mean()
        if rsi_avg > 0.5:
            print(f"👉 RSI > 70 is a MOMENTUM signal here. Price tends to keep rising (+{rsi_avg:.1f}%). Don't sell too early.")
        elif rsi_avg < -0.5:
            print(f"👉 RSI > 70 is a TRUE REVERSAL signal here. Price tends to drop ({rsi_avg:.1f}%). Take profit.")
        else:
            print(f"👉 RSI > 70 is NOISE. Price goes sideways.")

    # Check MACD Behavior
    macd_data = results_df[results_df['Signal'] == 'MACD Zero Cross Up (Bull)']
    if not macd_data.empty:
        macd_win = (macd_data['Return_5d'] > 0).mean()
        if macd_win < 0.5:
            print(f"👉 CAUTION: MACD Zero Cross Up often FAILS (Win Rate {macd_win*100:.0f}%). Wait for confirmation.")

# --- EXECUTION ---
user_input = input("Enter IHSG Ticker (e.g. BBCA, ANTM): ")
analyze_ihsg_behavior(user_input)
import yfinance as yf
import pandas as pd
import numpy as np
import sys
from scipy.stats import linregress

# --- CONFIGURATION ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format

class StockScanner:
    def __init__(self, ticker_input):
        self.ticker = self._format_ticker(ticker_input)
        self.df = None

    def _format_ticker(self, ticker_input):
        """Auto-formats for IHSG if needed."""
        t = ticker_input.upper()
        # Assume IHSG if 4 letters and not crypto/US format
        if len(t) == 4 and not t.endswith(".JK") and "-" not in t:
            return f"{t}.JK"
        return t

    def fetch_data(self, period="2y"):
        """Fetches data and calculates indicators."""
        print(f"\n--- 🇮🇩 Analyzing {self.ticker} ({period}) ---")
        try:
            self.df = yf.download(self.ticker, period=period, progress=False)
            
            # Handle MultiIndex (Fix for new yfinance versions)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)

            if self.df.empty:
                print(f"❌ Error: Data not found for {self.ticker}.")
                return False

            # Liquidity Check (Skip for Crypto/US)
            if ".JK" in self.ticker:
                avg_val = (self.df['Close'] * self.df['Volume']).tail(20).mean()
                if avg_val < 5_000_000_000:
                    print(f"⚠️  WARNING: Low Liquidity (Avg Val < 5M IDR). Signals may be unreliable.")

            self._calculate_indicators()
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def _calculate_indicators(self):
        df = self.df
        # Moving Averages
        df['MA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['MA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['MA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # MACD (12, 26, 9)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = df['EMA12'] - df['EMA26']
        
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    def run_backtest(self):
        """Retests historical signals."""
        if self.df is None: return

        df = self.df.copy()
        # Shift for crossover detection
        df['Prev_MA9'] = df['MA9'].shift(1)
        df['Prev_MA21'] = df['MA21'].shift(1)
        df['Prev_MA50'] = df['MA50'].shift(1)
        df['Prev_MACD'] = df['MACD_Line'].shift(1)
        df['Prev_RSI'] = df['RSI'].shift(1)

        signals = []
        
        # Loop through history (stop 20 days early for forward calc)
        for i in range(50, len(df) - 20):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Future Returns
            ret_5d = self._get_return(i, 5)
            ret_10d = self._get_return(i, 10)
            ret_20d = self._get_return(i, 20)
            
            sig_type = None

            # --- SIGNAL DEFINITIONS ---
            # 1. MA 9/21
            if prev['Prev_MA9'] < prev['Prev_MA21'] and curr['MA9'] > curr['MA21']:
                sig_type = 'MA 9 > 21 (Bull Momentum)'
            elif prev['Prev_MA9'] > prev['Prev_MA21'] and curr['MA9'] < curr['MA21']:
                sig_type = 'MA 9 < 21 (Bear Momentum)'
            
            # 2. MA 21/50
            elif prev['Prev_MA21'] < prev['Prev_MA50'] and curr['MA21'] > curr['MA50']:
                sig_type = 'MA 21 > 50 (Bull Trend)'
            elif prev['Prev_MA21'] > prev['Prev_MA50'] and curr['MA21'] < curr['MA50']:
                sig_type = 'MA 21 < 50 (Bear Trend)'

            # 3. MACD Zero
            elif prev['Prev_MACD'] < 0 and curr['MACD_Line'] > 0:
                sig_type = 'MACD Zero Cross Up'
            elif prev['Prev_MACD'] > 0 and curr['MACD_Line'] < 0:
                sig_type = 'MACD Zero Cross Down'

            # 4. RSI
            elif prev['Prev_RSI'] < 70 and curr['RSI'] >= 70:
                sig_type = 'RSI Enter >70 (Overbought)'
            elif prev['Prev_RSI'] > 30 and curr['RSI'] <= 30:
                sig_type = 'RSI Enter <30 (Oversold)'

            if sig_type:
                signals.append({
                    'Signal': sig_type,
                    'Ret_5d': ret_5d, 'Ret_10d': ret_10d, 'Ret_20d': ret_20d
                })

        # Summarize
        if not signals:
            print("No signals found in this period.")
            return

        res = pd.DataFrame(signals)
        summary = res.groupby('Signal').agg(
            Count=('Ret_5d', 'count'),
            Avg_5d=('Ret_5d', 'mean'),
            Avg_20d=('Ret_20d', 'mean'),
            Win_Rate_20d=('Ret_20d', lambda x: (x > 0).mean() * 100)
        ).sort_values(by='Count', ascending=False)

        print(f"\n📊 HISTORICAL BEHAVIOR (Last 2 Years)")
        print(summary)

    def run_prediction(self):
        """Predicts future crossovers based on current slope."""
        if self.df is None: return

        print(f"\n🔮 FUTURE PREDICTION (Based on current velocity)")
        
        # Helper to calc slope and intersection
        def predict_cross(short_ma, long_ma, name_short, name_long):
            # Get last 5 days
            s_series = self.df[short_ma].tail(5)
            l_series = self.df[long_ma].tail(5)
            
            # Linear Regression on slopes
            x = np.arange(len(s_series))
            slope_s, _, _, _, _ = linregress(x, s_series)
            slope_l, _, _, _, _ = linregress(x, l_series)
            
            curr_s = s_series.iloc[-1]
            curr_l = l_series.iloc[-1]
            gap = curr_s - curr_l
            net_slope = slope_s - slope_l # How fast gap is closing

            status = "Diverging (No Cross)"
            days = "N/A"
            
            # Logic: If Gap is + and Slope is -, they are converging
            if (gap > 0 and net_slope < 0) or (gap < 0 and net_slope > 0):
                d = abs(gap / net_slope)
                status = f"Converging in ~{d:.1f} days"
            
            trend = "BULLISH" if gap > 0 else "BEARISH"
            print(f"   • {name_short} vs {name_long}: Currently {trend}. {status}")

        predict_cross('MA9', 'MA21', 'MA 9', 'MA 21')
        predict_cross('MA21', 'MA50', 'MA 21', 'MA 50')
        
        # Current Price Context
        last_price = self.df['Close'].iloc[-1]
        last_rsi = self.df['RSI'].iloc[-1]
        print(f"   • Current Price: {last_price:,.0f} | RSI: {last_rsi:.1f}")

    def _get_return(self, idx, days):
        try:
            future = self.df.iloc[idx + days]['Close']
            curr = self.df.iloc[idx]['Close']
            return ((future - curr) / curr) * 100
        except IndexError:
            return np.nan

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = input("Enter Ticker (e.g. BBCA): ")

    scanner = StockScanner(ticker)
    if scanner.fetch_data():
        scanner.run_backtest()
        scanner.run_prediction()
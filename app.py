import streamlit as st
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import time

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(layout="wide", page_title="AI Market Cockpit Pro", page_icon="🛸")

# --- CUSTOM CSS FOR "PREMIUM" LOOK ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
try:
    GOOG_API_KEY = st.secrets["GOOG_API_KEY"]
    genai.configure(api_key=GOOG_API_KEY)
except Exception as e:
    st.error("API Key Error: Please check your Streamlit Secrets.")

# --- MATH ENGINE ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_vwap(df):
    v = df['Volume'].values
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return df.assign(VWAP=(tp * v).cumsum() / v.cumsum())

# --- DATA FETCHING ---
def get_daily_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # UPGRADED TO 1-HOUR CHART FOR HYPER-SENSITIVITY
        df = stock.history(period="1mo", interval="1h")
        if df.empty: return None

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        trend = df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]

        df['RSI'] = calculate_rsi(df['Close'])
        rsi = df['RSI'].iloc[-1]
        mom = rsi > 50

        df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()
        vol_today = df['Volume'].iloc[-1] > df['Vol_SMA'].iloc[-1]
        vol_yesterday = df['Volume'].iloc[-2] > df['Vol_SMA'].iloc[-2]
        vol = vol_today or vol_yesterday

        macd_line, sig_line = calculate_macd(df['Close'])
        macd_bull = macd_line.iloc[-1] > sig_line.iloc[-1]
        
        return {
            "Price": df['Close'].iloc[-1],
            "Trend": "🟢 UPTREND" if trend else "🔴 DOWNTREND",
            "Trend_Raw": trend,
            "RSI": f"{rsi:.1f}",
            "RSI_Raw": rsi,
            "Vol": "🟢 HIGH" if vol else "🔴 LOW",
            "MACD": "🟢 BULL" if macd_bull else "🔴 BEAR",
            "Score": sum([trend, mom, vol, macd_bull]),
            "News": stock.news
        }
    except: return None

def run_backtest(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y") # 2 years of data for a solid sample size
    if df.empty: return None

    # Calculate indicators for the entire 2-year dataframe
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Trend'] = df['Close'] > df['SMA_20']
    
    df['RSI'] = calculate_rsi(df['Close'])
    df['Mom'] = df['RSI'] > 50
    
    df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Vol'] = (df['Volume'] > df['Vol_SMA']) | (df['Volume'].shift(1) > df['Vol_SMA'].shift(1))
    
    macd_line, sig_line = calculate_macd(df['Close'])
    df['MACD_Bull'] = macd_line > sig_line
    
    # Calculate the Tech Score for every single day in history
    df['Tech_Score'] = df['Trend'].astype(int) + df['Mom'].astype(int) + df['Vol'].astype(int) + df['MACD_Bull'].astype(int)
    
    # Calculate Forward Returns (Look 5 and 10 days into the future)
    df['Return_5D'] = df['Close'].shift(-5) / df['Close'] - 1
    df['Return_10D'] = df['Close'].shift(-10) / df['Close'] - 1
    
    df = df.dropna() # Clean up recent days that don't have future data yet
    
    # Group by the Tech Score to get the win rates
    stats = df.groupby('Tech_Score').agg(
        Occurrences=('Close', 'count'),
        Win_Rate_5D=('Return_5D', lambda x: (x > 0).mean() * 100),
        Avg_Return_5D=('Return_5D', lambda x: x.mean() * 100),
        Win_Rate_10D=('Return_10D', lambda x: (x > 0).mean() * 100),
        Avg_Return_10D=('Return_10D', lambda x: x.mean() * 100)
    ).round(2)
    
    return stats

def get_ai_master_analysis(ticker, daily, micro):
    # Upgraded to Gemini 2.5 Pro for advanced reasoning
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    headlines = []
    if daily['News']:
        for n in daily['News'][:3]:
            title = n.get('title') or n.get('content', {}).get('title')
            if title: headlines.append(f"- {title}")
    
    prompt = f"""
    Act as an elite Hedge Fund Manager and Quantitative Analyst. Ticker: {ticker}
    LIVE PRICE: ${micro['Current_Price']:.2f}
    
    1. MACRO (Hourly Chart): Trend: {daily['Trend']}, Tech Score: {daily['Score']}/4
    2. MICRO (5m Chart): VWAP Level: ${micro['VWAP_Price']:.2f} ({micro['VWAP_Signal']}), RSI: {micro['RSI_5m']}
    3. NEWS: {str(headlines)}
    
    YOUR TASK:
    Analyze the data above, specifically breaking down what the Tech Score ({daily['Score']}/4) means for the current momentum.
    Calculate your targets strictly based on the LIVE PRICE of ${micro['Current_Price']:.2f}.
    
    Provide a definitive execution plan formatted EXACTLY like this:

    **Tech Score Breakdown:** (1 short sentence explaining the strength or weakness of the {daily['Score']}/4 score)

    **⚡ Day Trader Signal:** (BUY / SELL / HOLD)
    - Target: (Calculate a logical price target from the Live Price)
    - Stop Loss: (Calculate a strict stop loss from the Live Price)
    - Reasoning: (1 short sentence focusing on intraday momentum)

    **🛡️ Long-Term Signal:** (ACCUMULATE / REDUCE / HOLD)
    - Key Level: (Macro support/resistance to watch)
    - Reasoning: (1 short sentence focusing on macro trends and news)
    """
    for _ in range(3):
        try:
            return model.generate_content(prompt).text
        except Exception as e: 
            time.sleep(1)
            last_err = str(e)
    return f"⚠️ AI Connection Failed: {last_err}"

# --- UI LAYOUT ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🛸 AI Market Cockpit Pro")
with c2:
    if st.button("🔄 Refresh Data"):
        st.rerun()

with st.sidebar:
    st.header("⚙️ Control Panel")
    # ADDED THE BACKTEST ENGINE TO THE MENU HERE
    mode = st.radio("Select Mode", ["Single Ticker", "Market Scanner", "Backtest Engine"])
    st.info("Scanner checks: NVDA, AAPL, MSFT, TSLA, AMD, SPY, QQQ")

# --- MODE 1: SINGLE TICKER ---
if mode == "Single Ticker":
    ticker = st.text_input("Enter Ticker", value="NVDA").upper()
    
    daily = get_daily_data(ticker)
    micro = get_intraday_data(ticker)
    
    if daily and micro:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Live Price", f"${micro['Current_Price']:.2f}")
        col2.metric("VWAP Level", f"${micro['VWAP_Price']:.2f}", delta=micro['VWAP_Signal'])
        # UPDATED TO HOURLY TIMEFRAME
        col3.metric("Tech Score", f"{daily['Score']}/4", delta="Hourly Timeframe")
        col4.metric("5m Momentum", micro['RSI_5m'], delta=micro['RSI_Status'])
        
        st.markdown("### 🤖 AI Commander's Verdict")
        with st.container():
            verdict = get_ai_master_analysis(ticker, daily, micro)
            st.success(verdict)

        # --- NEW FEATURE: RISK MANAGEMENT ENGINE ---
        st.markdown("### ⚖️ Auto-Calculated Risk Levels")
        entry = micro['Current_Price']
        r1, r2, r3 = st.columns(3)
        
        if "BUY" in micro['VWAP_Signal']:
            sl = entry * 0.99
            tp1 = entry * 1.015
            tp2 = entry * 1.03
            r1.metric("🛑 Stop Loss (-1%)", f"${sl:.2f}")
            r2.metric("🎯 Take Profit 1 (+1.5%)", f"${tp1:.2f}")
            r3.metric("🚀 Take Profit 2 (+3%)", f"${tp2:.2f}")
        else:
            sl = entry * 1.01
            tp1 = entry * 0.985
            tp2 = entry * 0.97
            r1.metric("🛑 Short Stop (+1%)", f"${sl:.2f}")
            r2.metric("🎯 Cover Target 1 (-1.5%)", f"${tp1:.2f}")
            r3.metric("🚀 Cover Target 2 (-3%)", f"${tp2:.2f}")

        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        with col_left:
            with st.expander("📊 View Macro (Hourly) Details", expanded=False):
                st.write(f"**Trend:** {daily['Trend']}")
                st.write(f"**MACD:** {daily['MACD']}")
                st.write(f"**Volume:** {daily['Vol']}")
                safe_rsi = max(0.0, min(daily['RSI_Raw'] / 100.0, 1.0))
                st.progress(safe_rsi, text=f"RSI Strength: {daily['RSI']}")
        
        with col_right:
            with st.expander("🎯 View Micro (5-Min) Details", expanded=False):
                st.write(f"**VWAP Signal:** {micro['VWAP_Signal']}")
                st.write(f"**Current RSI:** {micro['RSI_5m']}")
                st.info("If Price > VWAP, institutions are buying.")
                
        with st.expander("📰 Read Latest News"):
            for n in daily['News'][:5]:
                title = n.get('title') or n.get('content', {}).get('title')
                if title:
                    st.markdown(f"- {title}")

# --- MODE 2: INSTITUTIONAL SCANNER ---
elif mode == "Market Scanner":
    st.subheader("📡 Institutional Market Scanner")
    st.caption("Scan entire sectors for high-probability setups.")
    
    # 1. Sector Selection
    col1, col2 = st.columns([2, 1])
    with col1:
        sector = st.selectbox("Select Sector to Scan", [
            "Mega-Cap Tech",
            "Semiconductors",
            "Crypto & Blockchain",
            "Custom List"
        ])
    
    with col2:
        if sector == "Mega-Cap Tech":
            scan_list = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
        elif sector == "Semiconductors":
            scan_list = ["AMD", "TSM", "AVGO", "QCOM", "INTC", "ASML", "MU"]
        elif sector == "Crypto & Blockchain":
            scan_list = ["MSTR", "COIN", "MARA", "RIOT", "HUT"]
        else:
            custom_input = st.text_input("Enter tickers (comma separated)", "SPY, QQQ, IWM")
            scan_list = [t.strip().upper() for t in custom_input.split(",")]

    if st.button("🚀 Run Deep Scan", use_container_width=True):
        scan_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(scan_list):
            status_text.text(f"Scanning {t}... ({i+1}/{len(scan_list)})")
            d = get_daily_data(t)
            m = get_intraday_data(t)
            
            if d and m:
                # Clean up emojis for the professional data table
                trend_clean = d['Trend'].replace("🟢 ", "").replace("🔴 ", "")
                macd_clean = d['MACD'].replace("🟢 ", "").replace("🔴 ", "")
                vwap_clean = m['VWAP_Signal'].replace("🟢 ", "").replace("🔴 ", "")
                
                scan_results.append({
                    "Ticker": t,
                    "Score": d['Score'],
                    "Price": f"${m['Current_Price']:.2f}",
                    "1H Trend": trend_clean,
                    "MACD": macd_clean,
                    "RSI (1H)": d['RSI'],
                    "Intraday VWAP": vwap_clean
                })
                
            progress_bar.progress((i + 1) / len(scan_list))
            
        status_text.text("Scan Complete!")
        
        if scan_results:
            # 2. Build the Interactive Data Table
            df_results = pd.DataFrame(scan_results)
            df_results = df_results.sort_values(by="Score", ascending=False).reset_index(drop=True)
            
            st.markdown("### 📊 Scan Results")
            st.dataframe(df_results, use_container_width=True)
            
            # 3. Quick AI Analysis Integration
            st.markdown("---")
            st.markdown("### 🤖 Quick AI Analysis")
            
            ai_col1, ai_col2 = st.columns([3, 1])
            with ai_col1:
                analyze_ticker = st.selectbox("Select a stock from the results to send to the AI Commander:", df_results['Ticker'].tolist())
            with ai_col2:
                st.write("") 
                st.write("")
                run_ai = st.button(f"Analyze {analyze_ticker}")
                
            if run_ai:
                with st.spinner(f"Commander analyzing {analyze_ticker}..."):
                    d_ai = get_daily_data(analyze_ticker)
                    m_ai = get_intraday_data(analyze_ticker)
                    verdict = get_ai_master_analysis(analyze_ticker, d_ai, m_ai)
                    st.success(verdict)
        else:
            st.warning("No data found for the selected tickers.")

# --- MODE 3: BACKTEST ENGINE ---
elif mode == "Backtest Engine":
    st.subheader("⏱️ Historical Accuracy Backtester")
    st.caption("Proving the Tech Score's win rate over the last 2 years.")
    
    test_ticker = st.text_input("Enter Ticker to Backtest", value="NVDA").upper()
    
    if st.button("Run 2-Year Backtest"):
        with st.spinner(f"Crunching 2 years of daily data for {test_ticker}..."):
            stats = run_backtest(test_ticker)
            
            if stats is not None and not stats.empty:
                st.markdown(f"### 📊 Tech Score Performance for {test_ticker}")
                st.write("This table tracks exactly what happened to the stock 5 days and 10 days AFTER it registered a specific Tech Score.")
                
                # Format the dataframe into a beautiful heat-mapped table
                st.dataframe(
                    stats.style.format("{:.2f}%", subset=['Win_Rate_5D', 'Avg_Return_5D', 'Win_Rate_10D', 'Avg_Return_10D'])
                               .background_gradient(cmap='RdYlGn', subset=['Win_Rate_5D', 'Win_Rate_10D']),
                    use_container_width=True
                )
                
                # Extract the 4/4 win rate for a dynamic insight
                if 4 in stats.index:
                    win_5 = stats.loc[4, 'Win_Rate_5D']
                    st.success(f"**Institutional Insight:** When {test_ticker} hits a perfect 4/4 Tech Score, historical data shows it has a **{win_5}% probability** of being profitable 5 days later.")
            else:
                st.error("Not enough data to run backtest.")

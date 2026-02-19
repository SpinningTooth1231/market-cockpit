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
# PASTE YOUR API KEY BELOW
# Fetch the key securely from Streamlit Secrets
GOOG_API_KEY = st.secrets["GOOG_API_KEY"]
try:
    genai.configure(api_key=GOOG_API_KEY)
except Exception as e:
    st.error(f"API Key Error: {e}")

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
        df = stock.history(period="6mo")
        if df.empty: return None

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        trend = df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]

        df['RSI'] = calculate_rsi(df['Close'])
        rsi = df['RSI'].iloc[-1]
        mom = 50 < rsi < 70

        df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()
        vol = df['Volume'].iloc[-1] > df['Vol_SMA'].iloc[-1]

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

def get_intraday_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d", interval="5m")
        if df.empty: return None

        df = calculate_vwap(df)
        df['RSI'] = calculate_rsi(df['Close'])
        
        last_close = df['Close'].iloc[-1]
        last_vwap = df['VWAP'].iloc[-1]
        last_rsi = df['RSI'].iloc[-1]
        
        vwap_signal = last_close > last_vwap
        
        return {
            "Current_Price": last_close,
            "VWAP_Price": last_vwap,
            "VWAP_Signal": "🟢 BUY" if vwap_signal else "🔴 SELL",
            "RSI_5m": f"{last_rsi:.1f}",
            "RSI_Status": "🔥 HOT" if last_rsi > 70 else "❄️ COLD" if last_rsi < 30 else "✅ OK"
        }
    except: return None

def get_ai_master_analysis(ticker, daily, micro):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    headlines = []
    if daily['News']:
        for n in daily['News'][:3]:
            title = n.get('title') or n.get('content', {}).get('title')
            if title: headlines.append(f"- {title}")
    
    prompt = f"""
    Act as a Hedge Fund Manager. Ticker: {ticker}
    1. MACRO (Daily Chart): Trend: {daily['Trend']}, Score: {daily['Score']}/4
    2. MICRO (5m Chart): VWAP Signal: {micro['VWAP_Signal']}, RSI: {micro['RSI_5m']}
    3. NEWS: {str(headlines)}
    
    YOUR TASK:
    Provide a "Sniper Execution Plan".
    - Verdict: (Buy Now / Wait for Dip / Short Sell)
    - Key Level: (Where to put Stop Loss)
    - Reasoning: (1 short sentence)
    """
    for _ in range(3):
        try:
            return model.generate_content(prompt).text
        except: time.sleep(1)
    return "⚠️ AI Busy. Try refreshing."

# --- UI LAYOUT ---
# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🛸 AI Market Cockpit Pro")
with c2:
    if st.button("🔄 Refresh Data"):
        st.rerun()

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    mode = st.radio("Select Mode", ["Single Ticker", "Market Scanner"])
    st.info("Scanner checks: NVDA, AAPL, MSFT, TSLA, AMD, SPY, QQQ")

# --- MODE 1: SINGLE TICKER ---
if mode == "Single Ticker":
    ticker = st.text_input("Enter Ticker", value="NVDA").upper()
    
    # Fetch Data
    daily = get_daily_data(ticker)
    micro = get_intraday_data(ticker)
    
    if daily and micro:
        # 1. HERO SECTION (The Big Numbers)
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Live Price", f"${micro['Current_Price']:.2f}")
        col2.metric("VWAP Level", f"${micro['VWAP_Price']:.2f}", delta=micro['VWAP_Signal'])
        col3.metric("Tech Score", f"{daily['Score']}/4", delta="Daily Timeframe")
        col4.metric("5m Momentum", micro['RSI_5m'], delta=micro['RSI_Status'])
        
        # 2. AI COMMANDER (The Decision)
        st.markdown("### 🤖 AI Commander's Verdict")
        with st.container(): # Grouping for visual separation
            verdict = get_ai_master_analysis(ticker, daily, micro)
            st.success(verdict)

        st.markdown("---")
        
        # 3. DEEP DIVE (Dropdowns for Details)
        col_left, col_right = st.columns(2)
        
        with col_left:
            with st.expander("📊 View Macro (Daily) Details", expanded=False):
                st.write(f"**Trend:** {daily['Trend']}")
                st.write(f"**MACD:** {daily['MACD']}")
                st.write(f"**Volume:** {daily['Vol']}")
                st.progress(daily['RSI_Raw']/100, text=f"RSI Strength: {daily['RSI']}")
        
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

# --- MODE 2: MARKET SCANNER ---
elif mode == "Market Scanner":
    st.subheader("🔍 Real-Time Opportunity Scanner")
    st.caption("Scanning top assets for 4/4 setups...")
    
    tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "SPY", "QQQ"]
    
    if st.button("Start Scan"):
        progress = st.progress(0)
        
        for i, t in enumerate(tickers):
            d = get_daily_data(t)
            progress.progress((i + 1) / len(tickers))
            
            if d:
                # Only show interesting stocks
                if d['Score'] >= 3:
                    with st.expander(f"🔥 {t} (Score: {d['Score']}/4) - CLICK TO EXPAND"):
                        m = get_intraday_data(t)
                        c1, c2 = st.columns(2)
                        c1.metric("Trend", d['Trend'])
                        c2.metric("Intraday Signal", m['VWAP_Signal'])
                        st.button(f"Analyze {t}", key=f"btn_{t}")
                elif d['Score'] <= 1:
                    with st.expander(f"🔻 {t} (Score: {d['Score']}/4) - WEAK"):
                        st.write("Bearish Setup. Be careful.")

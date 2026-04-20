import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# -------------------------------
# Load trained model and scaler
# -------------------------------
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# -------------------------------
# Custom CSS — Premium Dark Theme
# -------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Hide default Streamlit header & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hide sidebar collapse button — sidebar stays permanently open */
    [data-testid="collapseSidebarButton"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Force sidebar always visible and open */
    section[data-testid="stSidebar"] {
        transform: translateX(0px) !important;
        min-width: 244px !important;
        visibility: visible !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2a4a 40%, #0a3d62 70%, #00D4AA22 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 212, 170, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0,212,170,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00D4AA, #00B4D8, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #8899AA;
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1A1F2E, #151922);
        border: 1px solid rgba(0, 212, 170, 0.1);
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 212, 170, 0.35);
        box-shadow: 0 0 20px rgba(0, 212, 170, 0.08);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #6B7B8D;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FAFAFA;
    }
    .metric-value.up { color: #00D4AA; }
    .metric-value.down { color: #FF6B6B; }

    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(145deg, #1A1F2E, #151922);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .prediction-box.bullish {
        border-color: rgba(0, 212, 170, 0.4);
        box-shadow: 0 0 40px rgba(0, 212, 170, 0.08);
    }
    .prediction-box.bearish {
        border-color: rgba(255, 107, 107, 0.4);
        box-shadow: 0 0 40px rgba(255, 107, 107, 0.08);
    }
    .pred-label {
        font-size: 0.85rem;
        color: #6B7B8D;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .pred-price {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.3rem 0;
        letter-spacing: -1px;
    }
    .pred-price.up { color: #00D4AA; }
    .pred-price.down { color: #FF6B6B; }
    .pred-change {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.3rem;
    }
    .pred-change.up {
        background: rgba(0, 212, 170, 0.15);
        color: #00D4AA;
    }
    .pred-change.down {
        background: rgba(255, 107, 107, 0.15);
        color: #FF6B6B;
    }
    .pred-date {
        font-size: 0.8rem;
        color: #556677;
        margin-top: 0.8rem;
    }

    /* Trend Insight Card */
    .trend-card {
        background: linear-gradient(145deg, #1A1F2E, #151922);
        border-radius: 16px;
        padding: 2rem;s
        height: 100%;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .trend-title {
        font-size: 1rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 1rem;
    }
    .trend-indicator {
        font-size: 3.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .trend-text {
        font-size: 0.9rem;
        color: #8899AA;
        line-height: 1.6;
    }

    /* Info Card */
    .info-card {
        background: linear-gradient(145deg, #1A1F2E, #151922);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.5rem;
    }
    .info-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00D4AA;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .info-row:last-child { border-bottom: none; }
    .info-key {
        color: #6B7B8D;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .info-val {
        color: #FAFAFA;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0,212,170,0.2);
        display: inline-block;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #151922;
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 212, 170, 0.15) !important;
        color: #00D4AA !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151922 100%) !important;
        border-right: 1px solid rgba(0,212,170,0.1);
    }
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stDateInput label {
        color: #8899AA !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #00D4AA, #00B4D8) !important;
        color: #0E1117 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(0,212,170,0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* Footer */
    .custom-footer {
        background: #0a0e14;
        border-top: 1px solid rgba(0,212,170,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 3rem;
        text-align: center;
    }
    .footer-text {
        color: #445566;
        font-size: 0.78rem;
        line-height: 1.8;
    }
    .footer-accent {
        color: #00D4AA;
        font-weight: 600;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb {
        background: #1A1F2E;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #00D4AA; }

    /* Data table */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------

# Force sidebar open — clears any collapsed browser state
st.markdown("""
<script>
(function() {
    // Clear any stored collapsed state in localStorage
    try {
        Object.keys(localStorage).forEach(function(key) {
            if (key.includes('sidebar') || key.includes('Sidebar')) {
                localStorage.removeItem(key);
            }
        });
    } catch(e) {}
    // If sidebar is collapsed, click the expand button
    function openSidebar() {
        var btn = document.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                  document.querySelector('[data-testid="collapsedControl"] button');
        if (btn) { btn.click(); }
    }
    setTimeout(openSidebar, 500);
    setTimeout(openSidebar, 1500);
})();
</script>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <div style="font-size:2.5rem;">🧠</div>
        <div style="font-size:1.1rem; font-weight:700; color:#00D4AA; margin-top:0.3rem;">LSTM Predictor</div>
        <div style="font-size:0.75rem; color:#556677; margin-top:0.2rem;">Deep Learning Model</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="color:#00D4AA; font-weight:700; font-size:0.85rem; letter-spacing:1px;">🔍 STOCK SELECTION</p>', unsafe_allow_html=True)

    options = ["Add or Select a Stock", "RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "Custom..."]
    selected_option = st.selectbox("Stock Symbol", options)
    
    import streamlit.components.v1 as components
    
    if selected_option == "Custom...":
        stock_input = st.text_input("Enter custom symbol", value="", placeholder="e.g. WIPRO")
        
        # Inject JavaScript to auto-focus the specific input box
        components.html(
            """
            <script>
                // Find the specific text input by looking for its aria-label in the parent Streamlit DOM
                const inputElement = window.parent.document.querySelector('input[aria-label="Enter custom symbol"]');
                if (inputElement) {
                    inputElement.focus();
                }
            </script>
            """,
            height=0,
        )
    elif selected_option == "Add or Select a Stock":
        stock_input = "RELIANCE" # Safe default
    else:
        stock_input = selected_option
    
    # Automatically append .NS if not provided by the user
    stock_symbol = f"{stock_input.upper()}.NS" if not stock_input.upper().endswith(".NS") else stock_input.upper()

    st.markdown('<p style="color:#00D4AA; font-weight:700; font-size:0.85rem; letter-spacing:1px; margin-top:1rem;">📅 DATE RANGE</p>', unsafe_allow_html=True)

    start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Run Prediction")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; margin-top:1rem;">
        <div style="font-size:0.7rem; color:#445566; line-height:1.6;">
            <strong style="color:#6B7B8D;">Popular Symbols</strong><br>
            RELIANCE • TCS<br>
            INFY • HDFCBANK<br>
            WIPRO • SBIN
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Initialize session state
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "next_day_price" not in st.session_state:
    st.session_state.next_day_price = None
if "predicted_prices" not in st.session_state:
    st.session_state.predicted_prices = None
if "y_test_actual" not in st.session_state:
    st.session_state.y_test_actual = None
if "stock_info" not in st.session_state:
    st.session_state.stock_info = None

# -------------------------------
# Fetch Data & Calculate Predictions
# (Core logic UNCHANGED)
# -------------------------------
if predict_btn:
    with st.spinner("🔄 Fetching market data & running LSTM prediction..."):
        df = yf.download(stock_symbol, start=start_date, end=end_date)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" in df.columns:
            df = df.dropna(subset=["Close"]).copy()

        if df.empty:
            st.error("❌ No valid closing-price data found for this stock symbol in the selected range.")
        elif len(df) < 60:
            st.error("❌ Not enough valid data points after cleanup. Please choose a wider date range.")
        else:
            st.session_state.df = df

            # ---- Next-day prediction (UNCHANGED) ----
            last_60_days = df['Close'][-60:].values.reshape(-1,1)
            scaled_last_60 = scaler.transform(last_60_days)
            X_input = scaled_last_60.reshape(1,60,1)
            next_day_pred = model.predict(X_input)
            st.session_state.next_day_price = scaler.inverse_transform(next_day_pred)[0][0]

            # ---- Historical predictions (UNCHANGED) ----
            scaled_data_full = scaler.transform(df['Close'].values.reshape(-1,1))
            X_test, y_test = [], []
            for i in range(60, len(scaled_data_full)):
                X_test.append(scaled_data_full[i-60:i,0])
                y_test.append(scaled_data_full[i,0])
            X_test, y_test = np.array(X_test), np.array(y_test)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)

            st.session_state.predicted_prices = scaler.inverse_transform(model.predict(X_test))
            st.session_state.y_test_actual = scaler.inverse_transform(np.array(y_test).reshape(-1,1))

            # ---- Fetch stock info ----
            try:
                ticker = yf.Ticker(stock_symbol)
                st.session_state.stock_info = ticker.info
            except:
                st.session_state.stock_info = {}

# -------------------------------
# Hero Banner (Dynamic)
# -------------------------------
if st.session_state.df is not None and st.session_state.stock_info:
    info = st.session_state.stock_info
    company_name = info.get('longName', info.get('shortName', stock_symbol))
    sector = info.get('sector', '')
    sector_display = f" • {sector}" if sector and sector != 'N/A' else ""
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">📈 {company_name}</div>
        <p class="hero-subtitle">{stock_symbol}{sector_display} — LSTM Next-Day Price Forecast</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📈 Stock Price Prediction</div>
        <p class="hero-subtitle">Powered by LSTM Deep Learning — Next-day closing price forecast for NSE stocks</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Helper: Format large numbers
# -------------------------------
def format_number(num):
    if num is None:
        return "N/A"
    if num >= 1e12:
        return f"₹{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"₹{num/1e9:.2f}B"
    elif num >= 1e7:
        return f"₹{num/1e7:.2f}Cr"
    elif num >= 1e5:
        return f"₹{num/1e5:.2f}L"
    else:
        return f"₹{num:,.2f}"

# -------------------------------
# Display — Tabbed Layout
# -------------------------------
if st.session_state.df is not None:

    # ✅ Ensure all numeric values are scalar floats (UNCHANGED logic)
    last_close = float(st.session_state.df['Close'].iloc[-1])
    next_price = float(st.session_state.next_day_price)
    change = float(next_price - last_close)
    percent_change = (change / last_close) * 100
    next_date = (st.session_state.df.index[-1] + pd.Timedelta(days=1)).strftime("%d %b %Y")
    is_bullish = change >= 0

    # ========================
    # KPI Metric Cards Row
    # ========================
    df = st.session_state.df
    latest_open = float(df['Open'].iloc[-1])
    latest_high = float(df['High'].iloc[-1])
    latest_low = float(df['Low'].iloc[-1])
    latest_close = float(df['Close'].iloc[-1])
    latest_volume = int(df['Volume'].iloc[-1])

    kpi_cols = st.columns(5)
    kpi_data = [
        ("Open", f"₹{latest_open:,.2f}"),
        ("High", f"₹{latest_high:,.2f}"),
        ("Low", f"₹{latest_low:,.2f}"),
        ("Close", f"₹{latest_close:,.2f}"),
        ("Volume", f"{latest_volume:,}"),
    ]
    for col, (label, value) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================
    # Tabs
    # ========================
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Charts", "📋 Stock Info"])

    # ========================
    # TAB 1 — Prediction
    # ========================
    with tab1:
        col1, col2 = st.columns([3, 2], gap="large")

        with col1:
            direction = "bullish" if is_bullish else "bearish"
            arrow = "▲" if is_bullish else "▼"
            color_class = "up" if is_bullish else "down"
            emoji = "📈" if is_bullish else "📉"
            sign = "+" if is_bullish else ""

            st.markdown(f"""
            <div class="prediction-box {direction}">
                <div class="pred-label">Predicted Closing Price — {next_date}</div>
                <div class="pred-price {color_class}">₹{next_price:,.2f}</div>
                <div class="pred-change {color_class}">
                    {arrow} {sign}{change:,.2f} ({sign}{percent_change:.2f}%) {emoji}
                </div>
                <div class="pred-date">
                    Last Close: ₹{last_close:,.2f} • Symbol: {stock_symbol}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            trend_emoji = "📈" if is_bullish else "📉"
            trend_text = (
                "The LSTM model predicts an <strong style='color:#00D4AA;'>upward movement</strong> in price. "
                "This suggests potential buying momentum based on the last 60 days of price action."
            ) if is_bullish else (
                "The LSTM model predicts a <strong style='color:#FF6B6B;'>downward movement</strong> in price. "
                "This suggests a possible correction or selling pressure based on recent trends."
            )
            st.markdown(f"""
            <div class="trend-card">
                <div class="trend-title">🧠 Trend Analysis</div>
                <div class="trend-indicator">{trend_emoji}</div>
                <div class="trend-text">{trend_text}</div>
                <br>
                <div style="font-size:0.75rem; color:#445566;">
                    ⚠️ Based on historical patterns. Not financial advice.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ---- Prediction vs Actual Chart ----
        st.markdown('<div class="section-header">📈 Predicted vs Actual — Last 100 Days</div>', unsafe_allow_html=True)

        last_n = 100
        y_actual = st.session_state.y_test_actual[-last_n:].flatten()
        y_pred = st.session_state.predicted_prices[-last_n:].flatten()
        next_p = float(st.session_state.next_day_price)

        fig_pred = go.Figure()

        fig_pred.add_trace(go.Scatter(
            x=list(range(last_n)),
            y=y_actual,
            name="Actual Price",
            line=dict(color="#00B4D8", width=2.5),
            hovertemplate="Day %{x}<br>Actual: ₹%{y:,.2f}<extra></extra>"
        ))

        fig_pred.add_trace(go.Scatter(
            x=list(range(last_n)),
            y=y_pred,
            name="Predicted Price",
            line=dict(color="#00D4AA", width=2.5),
            hovertemplate="Day %{x}<br>Predicted: ₹%{y:,.2f}<extra></extra>"
        ))

        # Next day projection
        fig_pred.add_trace(go.Scatter(
            x=[last_n - 1, last_n],
            y=[y_pred[-1], next_p],
            name="Next Day Projection",
            line=dict(color="#FFD700", width=2.5, dash="dot"),
            hovertemplate="Next Day<br>Projected: ₹%{y:,.2f}<extra></extra>"
        ))

        # Next day marker
        fig_pred.add_trace(go.Scatter(
            x=[last_n],
            y=[next_p],
            name="Next Day Forecast",
            mode="markers",
            marker=dict(size=12, color="#FFD700", symbol="star", line=dict(width=2, color="#FFF")),
            hovertemplate="⭐ Next Day<br>Forecast: ₹%{y:,.2f}<extra></extra>"
        ))

        fig_pred.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=11)
            ),
            xaxis=dict(title="Days", gridcolor="rgba(255,255,255,0.05)", showgrid=True),
            yaxis=dict(title="Price (₹)", gridcolor="rgba(255,255,255,0.05)", showgrid=True),
            hovermode="x unified"
        )

        st.plotly_chart(fig_pred, use_container_width=True)

    # ========================
    # TAB 2 — Charts
    # ========================
    with tab2:
        st.markdown('<div class="section-header">🕯️ Candlestick Chart with Volume</div>', unsafe_allow_html=True)

        chart_df = st.session_state.df.copy()
        # Flatten multi-level columns if present
        if hasattr(chart_df.columns, 'levels'):
            chart_df.columns = chart_df.columns.get_level_values(0)

        # Calculate SMAs
        chart_df['SMA_20'] = chart_df['Close'].rolling(window=20).mean()
        chart_df['SMA_50'] = chart_df['Close'].rolling(window=50).mean()

        # Candlestick + Volume subplot
        fig_candle = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=None
        )

        # Candlestick
        fig_candle.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name="OHLC",
            increasing_line_color="#00D4AA",
            decreasing_line_color="#FF6B6B",
            increasing_fillcolor="#00D4AA",
            decreasing_fillcolor="#FF6B6B",
        ), row=1, col=1)

        # SMA 20
        fig_candle.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df['SMA_20'],
            name="SMA 20",
            line=dict(color="#00B4D8", width=1.5),
            hovertemplate="SMA 20: ₹%{y:,.2f}<extra></extra>"
        ), row=1, col=1)

        # SMA 50
        fig_candle.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df['SMA_50'],
            name="SMA 50",
            line=dict(color="#FFD700", width=1.5),
            hovertemplate="SMA 50: ₹%{y:,.2f}<extra></extra>"
        ), row=1, col=1)

        # Volume bars
        colors = ['#00D4AA' if c >= o else '#FF6B6B'
                  for c, o in zip(chart_df['Close'], chart_df['Open'])]
        fig_candle.add_trace(go.Bar(
            x=chart_df.index,
            y=chart_df['Volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.5,
            hovertemplate="Vol: %{y:,}<extra></extra>"
        ), row=2, col=1)

        fig_candle.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=600,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=11)
            ),
            hovermode="x unified"
        )

        fig_candle.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
        fig_candle.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        fig_candle.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig_candle.update_yaxes(title_text="Volume", row=2, col=1)

        st.plotly_chart(fig_candle, use_container_width=True)

        # ---- Closing Price Trend ----
        st.markdown('<div class="section-header">📈 Closing Price Trend</div>', unsafe_allow_html=True)

        fig_close = go.Figure()
        fig_close.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df['Close'],
            name="Close",
            fill='tozeroy',
            fillcolor='rgba(0,212,170,0.08)',
            line=dict(color='#00D4AA', width=2),
            hovertemplate="%{x|%d %b %Y}<br>Close: ₹%{y:,.2f}<extra></extra>"
        ))

        fig_close.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
            yaxis=dict(title="Price (₹)", gridcolor="rgba(255,255,255,0.05)"),
            hovermode="x unified"
        )

        st.plotly_chart(fig_close, use_container_width=True)

    # ========================
    # TAB 3 — Stock Info
    # ========================
    with tab3:
        info = st.session_state.stock_info or {}

        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            company_name = info.get('longName', info.get('shortName', stock_symbol))
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            market_cap = info.get('marketCap', None)
            pe_ratio = info.get('trailingPE', None)
            dividend = info.get('dividendYield', None)

            pe_display = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
            div_display = f"{dividend*100:.2f}%" if dividend else "N/A"

            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-title">🏢 Company Overview</div>
                <div class="info-row"><span class="info-key">Company</span><span class="info-val">{company_name}</span></div>
                <div class="info-row"><span class="info-key">Sector</span><span class="info-val">{sector}</span></div>
                <div class="info-row"><span class="info-key">Industry</span><span class="info-val">{industry}</span></div>
                <div class="info-row"><span class="info-key">Market Cap</span><span class="info-val">{format_number(market_cap)}</span></div>
                <div class="info-row"><span class="info-key">P/E Ratio</span><span class="info-val">{pe_display}</span></div>
                <div class="info-row"><span class="info-key">Dividend Yield</span><span class="info-val">{div_display}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            week_high = info.get('fiftyTwoWeekHigh', None)
            week_low = info.get('fiftyTwoWeekLow', None)
            avg_vol = info.get('averageVolume', None)
            prev_close = info.get('previousClose', None)
            day_high = info.get('dayHigh', None)
            day_low = info.get('dayLow', None)

            wh = f"₹{week_high:,.2f}" if week_high else "N/A"
            wl = f"₹{week_low:,.2f}" if week_low else "N/A"
            av = f"{avg_vol:,}" if avg_vol else "N/A"
            pc = f"₹{prev_close:,.2f}" if prev_close else "N/A"
            dh = f"₹{day_high:,.2f}" if day_high else "N/A"
            dl = f"₹{day_low:,.2f}" if day_low else "N/A"

            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-title">📊 Market Statistics</div>
                <div class="info-row"><span class="info-key">52-Week High</span><span class="info-val">{wh}</span></div>
                <div class="info-row"><span class="info-key">52-Week Low</span><span class="info-val">{wl}</span></div>
                <div class="info-row"><span class="info-key">Prev Close</span><span class="info-val">{pc}</span></div>
                <div class="info-row"><span class="info-key">Day High</span><span class="info-val">{dh}</span></div>
                <div class="info-row"><span class="info-key">Day Low</span><span class="info-val">{dl}</span></div>
                <div class="info-row"><span class="info-key">Avg Volume</span><span class="info-val">{av}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # ---- Historical Data Table ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📋 Historical Price Data</div>', unsafe_allow_html=True)

        display_df = st.session_state.df.tail(50).copy()
        if hasattr(display_df.columns, 'levels'):
            display_df.columns = display_df.columns.get_level_values(0)
        display_df = display_df[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        display_df = display_df.sort_index(ascending=False)
        display_df.index = display_df.index.strftime('%d %b %Y')
        display_df.index.name = "Date"

        st.dataframe(display_df, use_container_width=True, height=400)

    # ========================
    # Footer
    # ========================
    st.markdown(f"""
    <div class="custom-footer">
        <div class="footer-text">
            ⚠️ <strong style="color:#6B7B8D;">Disclaimer:</strong> This tool is for <span class="footer-accent">educational purposes only</span>.
            Predictions are based on historical data patterns and should not be used as financial advice.<br>
            ⏳ Built with ❤️ using <span class="footer-accent">LSTM Deep Learning</span><br>
            👨‍💻 Developed by <span class="footer-accent">Mohd Ashar Ansari</span>, <span class="footer-accent">Mohd Ahmad</span> & <span class="footer-accent">Mohd Amaan Siddiqui</span><br>
            <span style="color:#334455;">Last updated: {datetime.now().strftime("%d %b %Y, %I:%M %p")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ---- Empty State ----
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">🧠</div>
        <div style="font-size:1.3rem; font-weight:600; color:#6B7B8D; margin-bottom:0.5rem;">
            Select a stock & click <span style="color:#00D4AA;">Run Prediction</span>
        </div>
        <div style="font-size:0.9rem; color:#445566;">
            Enter an NSE stock symbol in the sidebar and set your date range to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

# app.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------------------
# Load trained model and scaler
# -------------------------------
model = load_model('lstm_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")
st.write("Predict next-day closing price of any NSE stock using an LSTM model.")

# Sidebar input
st.sidebar.header("🔍 Stock Selection")
stock_symbol = st.sidebar.text_input("Enter NSE Stock Symbol (e.g. RELIANCE.NS, TCS.NS)", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

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

# -------------------------------
# Fetch Data & Calculate Predictions
# -------------------------------
if st.button("Predict"):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found for this stock symbol. Try another one (e.g., INFY.NS, TCS.NS).")
    else:
        st.session_state.df = df

        # ---- Next-day prediction ----
        last_60_days = df['Close'][-60:].values.reshape(-1,1)
        scaled_last_60 = scaler.transform(last_60_days)
        X_input = scaled_last_60.reshape(1,60,1)
        next_day_pred = model.predict(X_input)
        st.session_state.next_day_price = scaler.inverse_transform(next_day_pred)[0][0]

        # ---- Historical predictions ----
        scaled_data_full = scaler.transform(df['Close'].values.reshape(-1,1))
        X_test, y_test = [], []
        for i in range(60, len(scaled_data_full)):
            X_test.append(scaled_data_full[i-60:i,0])
            y_test.append(scaled_data_full[i,0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)

        st.session_state.predicted_prices = scaler.inverse_transform(model.predict(X_test))
        st.session_state.y_test_actual = scaler.inverse_transform(np.array(y_test).reshape(-1,1))

# -------------------------------
# Display Data & Predictions
# -------------------------------
if st.session_state.df is not None:
    # ✅ Ensure all numeric values are scalar floats
    last_close = float(st.session_state.df['Close'].iloc[-1])
    next_price = float(st.session_state.next_day_price)
    change = float(next_price - last_close)
    percent_change = (change / last_close) * 100
    next_date = (st.session_state.df.index[-1] + pd.Timedelta(days=1)).strftime("%d %b %Y")

    st.markdown("## 🔮 Next-Day Stock Price Prediction")
    col1, col2 = st.columns(2)

    with col1:
        if change >= 0:
            st.metric(
                label=f"{stock_symbol} Predicted Close ({next_date})",
                value=f"₹{next_price:.2f}",
                delta=f"+{percent_change:.2f}% 📈"
            )
        else:
            st.metric(
                label=f"{stock_symbol} Predicted Close ({next_date})",
                value=f"₹{next_price:.2f}",
                delta=f"{percent_change:.2f}% 📉"
            )

    with col2:
        st.write("**Trend Insight:**")
        if change >= 0:
            st.success("Model predicts an upward movement 📈 — possible gain ahead.")
        else:
            st.error("Model predicts a slight downward movement 📉 — possible correction.")

    # ---- Checkbox for historical chart ----
    if st.checkbox("📈 Show Predicted vs Actual (Last 100 Days + Next Day)"):
        last_n = 100
        y_actual = st.session_state.y_test_actual[-last_n:]
        y_pred = st.session_state.predicted_prices[-last_n:]
        next_price = float(st.session_state.next_day_price)

        fig, ax = plt.subplots(figsize=(12,6))

        x_range = range(last_n)  # x-axis for last 100 days

        # Plot actual and predicted lines for last 100 days
        ax.plot(x_range, y_actual, color='blue', label='Actual Price', linewidth=2)
        ax.plot(x_range, y_pred, color='orange', label='Predicted Price', linewidth=2)

        # Extend predicted line for next day as dotted
        extended_x = [last_n-1, last_n]
        extended_y = [y_pred[-1].item(), next_price]  # continue from last predicted point
        ax.plot(extended_x, extended_y, color='orange', linestyle='--', linewidth=2, label='Next Day Projection')

        # Styling
        ax.set_title(f"{stock_symbol} - Actual vs Predicted (Last {last_n} Days + Next Day)", fontsize=14, weight='bold')
        ax.set_xlabel("Days", fontsize=12)
        ax.set_ylabel("Price (INR)", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

        st.pyplot(fig)
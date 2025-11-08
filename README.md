📈 Stock Price Prediction using LSTM (Streamlit App)

🔍 Overview

This project predicts the next-day closing price of a selected stock using a Long Short-Term Memory (LSTM) neural network model.
It is built with Streamlit, TensorFlow, and yfinance, providing an interactive and educational demonstration of time-series forecasting using deep learning.

Developed as part of a college mini-project to showcase the application of machine learning in financial forecasting.

🧠 Key Features

📊 Real-time Stock Data — fetched automatically using the yfinance API.

🔮 Next-Day Price Prediction — predicts the upcoming day’s closing price.

🧾 Pre-trained LSTM Model — trained on historical closing price data.

⚙️ Interactive Web Interface — built using Streamlit.

📉 Trend Insights — visually represents model predictions vs. actual data.

📈 Automatic Scaling — MinMaxScaler applied for consistent normalization.

🧩 Project Structure

Stock_prediction_college/
│
├── app.py                  # Streamlit app (main file)
├── Stock_notebook.ipynb    # Model training notebook
├── lstm_model.h5           # Trained LSTM model
├── scaler.pkl              # Saved MinMaxScaler object
├── requirements.txt        # Dependencies
├── .gitignore              # Ignore unnecessary files (venv, checkpoints, etc.)
└── README.md               # Project documentation

🚀 How to Run the Project

1. Clone the Repository

git clone https://github.com/LetsCodewithAshar/Stock_Price_Prediction_LSTM.git
cd Stock_Price_Prediction_LSTM

2. Create and Activate a Virtual Environment

python -m venv venv
venv\Scripts\activate      # For Windows

# or

source venv/bin/activate   # For macOS/Linux

3. Install Dependencies

pip install -r requirements.txt

4. Run Streamlit App

streamlit run app.py

📂 Input and Output

Input:

Stock symbol (e.g., INFY.NS, TCS.NS, AAPL)

Start and End Dates

Output:

Predicted next-day closing price

Price movement direction (📈 Up / 📉 Down)

Comparison chart of actual vs predicted prices

🧠 Model Details

Model Type: LSTM (Long Short-Term Memory)

Training Data: Historical stock closing prices

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Prediction Output: Next-day closing price

🧩 Future Improvements

🧮 Add technical indicators (e.g., Moving Averages, RSI, MACD).

🌐 Include real-time live updates.

📅 Extend to multi-day forecasting.

🧠 Use ensemble deep learning models (GRU + LSTM).

💬 Add news sentiment analysis to improve accuracy.

👨‍💻 Team Members

Developed By:

🧑‍💻 Mohd Ashar Ansari

🧑‍💻 Mohd Ahmad

🧑‍💻 Mohd Amaan Siddiqui

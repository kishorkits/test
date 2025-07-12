import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Stock Price Prediction App")

ticker = st.text_input("Enter Stock Ticker Symbol (e.g. AAPL)", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if st.button("Predict"):
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found for this ticker.")
    else:
        st.write("### Historical Stock Prices", data.tail())

        data['Date'] = data.index
        data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)

        X = np.array(data['Date_ordinal']).reshape(-1, 1)
        y = np.array(data['Close'])

        model = LinearRegression()
        model.fit(X, y)

        future_date = pd.to_datetime(end_date + pd.Timedelta(days=30))
        future_ordinal = future_date.toordinal()
        predicted_price = model.predict([[future_ordinal]])

        st.write(f"### 📊 Predicted closing price on {future_date.date()} is **${predicted_price[0]:.2f}**")

        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(data['Date'], data['Close'], label="Historical Close Price")
        plt.scatter(future_date, predicted_price, color='red', label="Predicted Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.title(f"{ticker} Stock Price Prediction")
        plt.legend()
        st.pyplot(plt)

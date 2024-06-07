import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from ta.others import CCIIndicator

# Set the title and layout of the Streamlit app
st.set_page_config(layout="wide")
st.title("Comprehensive Trading Candlestick Chart for US Stock Indices")

# Sidebar for user inputs
st.sidebar.title("User Inputs")
indices = {
    'S&P 500': '^GSPC',
    'Dow Jones': '^DJI',
    'Nasdaq': '^IXIC'
}
selected_index = st.sidebar.selectbox("Select Index", list(indices.keys()))
ticker = indices[selected_index]
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
timeframe = st.sidebar.selectbox("Select Timeframe", ['1d', '1wk', '1mo'])

# Fetch data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date, interval=timeframe)
data.reset_index(inplace=True)

# Technical Indicators
def calculate_indicators(data):
    data["SMA_20"] = SMAIndicator(data["Close"], window=20).sma_indicator()
    data["SMA_50"] = SMAIndicator(data["Close"], window=50).sma_indicator()
    data["SMA_200"] = SMAIndicator(data["Close"], window=200).sma_indicator()
    data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()
    macd = MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Diff"] = macd.macd_diff()
    data["ATR"] = AverageTrueRange(data["High"], data["Low"], data["Close"]).average_true_range()
    bb = BollingerBands(data["Close"])
    data["BB_High"] = bb.bollinger_hband()
    data["BB_Mid"] = bb.bollinger_mavg()
    data["BB_Low"] = bb.bollinger_lband()
    data["OBV"] = OnBalanceVolumeIndicator(data["Close"], data["Volume"]).on_balance_volume()
    ichimoku = IchimokuIndicator(data["High"], data["Low"])
    data["Ichimoku_Conv"] = ichimoku.ichimoku_conversion_line()
    data["Ichimoku_Base"] = ichimoku.ichimoku_base_line()
    data["Ichimoku_Lead1"] = ichimoku.ichimoku_a()
    data["Ichimoku_Lead2"] = ichimoku.ichimoku_b()
    data["Stochastic"] = StochasticOscillator(data["Close"]).stoch()
    data["CCI"] = CCIIndicator(data["High"], data["Low"], data["Close"], window=20).cci()
    data["Pivot"] = (data["High"] + data["Low"] + data["Close"]) / 3
    return data

data = calculate_indicators(data)

# Plotting
def plot_data(data):
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data['Date'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlesticks'))
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], mode='lines', name='SMA 50'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], mode='lines', name='SMA 200'))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_High'], mode='lines', name='BB High'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Low'], mode='lines', name='BB Low'))
    
    # Ichimoku Cloud
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_Lead1'], mode='lines', name='Ichimoku Lead 1'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_Lead2'], mode='lines', name='Ichimoku Lead 2', fill='tonexty'))
    
    # Layout
    fig.update_layout(title=f"{selected_index} Price with Technical Indicators",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      xaxis_rangeslider_visible=False)
    
    return fig

# Display the data and plots
st.subheader("Price Data")
st.write(data)

st.subheader("Candlestick Chart with Technical Indicators")
st.plotly_chart(plot_data(data))

st.subheader("Technical Summary")
st.write("Buy/Sell signals based on technical indicators here...")

st.subheader("Momentum Oscillators")
st.write("RSI: ", data["RSI"].iloc[-1])

st.subheader("Trend Oscillators")
st.write("MACD: ", data["MACD"].iloc[-1])

st.subheader("Volatility Indicators")
st.write("ATR: ", data["ATR"].iloc[-1])

st.subheader("Pivot Points")
st.write("Pivot: ", data["Pivot"].iloc[-1])

st.subheader("Stochastic Oscillator")
st.write("Stochastic: ", data["Stochastic"].iloc[-1])

st.subheader("Commodity Channel Index (CCI)")
st.write("CCI: ", data["CCI"].iloc[-1])

st.subheader("On-balance Volume (OBV)")
st.write("OBV: ", data["OBV"].iloc[-1])

# Add additional sections as needed

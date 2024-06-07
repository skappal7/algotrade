import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import requests
from bs4 import BeautifulSoup

# Set the title and layout of the Streamlit app
st.set_page_config(layout="wide")
st.title("Algorithm-Driven Trading App for S&P 500")

# Sidebar for user inputs
st.sidebar.title("User Inputs")
ticker = st.sidebar.text_input("Enter Ticker", "^GSPC")  # S&P 500 ticker by default
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Fetch data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)
data.reset_index(inplace=True)

# Function to fetch news
def fetch_news(ticker):
    url = f'https://www.google.com/search?q={ticker}+stock+news'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find_all('div', attrs={'class': 'BNeawe vvjwJb AP7Wnd'})
    news = [headline.get_text() for headline in headlines[:5]]
    return news

# Function to calculate OBV
def calculate_obv(data):
    obv = [0]
    for i in range(1, len(data['Close'])):
        if data['Close'][i] > data['Close'][i - 1]:
            obv.append(obv[-1] + data['Volume'][i])
        elif data['Close'][i] < data['Close'][i - 1]:
            obv.append(obv[-1] - data['Volume'][i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=data.index)

# Technical Indicators
def calculate_indicators(data):
    data["SMA_20"] = SMAIndicator(data["Close"], window=20).sma_indicator()
    data["EMA_20"] = EMAIndicator(data["Close"], window=20).ema_indicator()
    data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()
    macd = MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Diff"] = macd.macd_diff()
    data["ATR"] = AverageTrueRange(data["High"], data["Low"], data["Close"]).average_true_range()
    data["BB_High"], data["BB_Mid"], data["BB_Low"] = BollingerBands(data["Close"]).bollinger_hband(), BollingerBands(data["Close"]).bollinger_mavg(), BollingerBands(data["Close"]).bollinger_lband()
    data["OBV"] = calculate_obv(data)
    ichimoku = IchimokuIndicator(data["High"], data["Low"])
    data["Ichimoku_Conv"] = ichimoku.ichimoku_conversion_line()
    data["Ichimoku_Base"] = ichimoku.ichimoku_base_line()
    data["Pivot"] = (data["High"] + data["Low"] + data["Close"]) / 3
    return data

data = calculate_indicators(data)

# Algorithmic Trading Strategies (simplified examples)
def mean_reversion_strategy(data):
    data['Signal'] = np.where(data['RSI'] < 30, 'Buy', np.where(data['RSI'] > 70, 'Sell', 'Hold'))
    return data

def moving_average_strategy(data):
    data['Signal'] = np.where(data['SMA_20'] > data['EMA_20'], 'Buy', 'Sell')
    return data

data = mean_reversion_strategy(data)
data = moving_average_strategy(data)

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
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_20'], mode='lines', name='EMA 20'))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_High'], mode='lines', name='BB High'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Low'], mode='lines', name='BB Low'))
    
    # Layout
    fig.update_layout(title="S&P 500 Price with Technical Indicators",
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

st.subheader("Trading Signals")
st.write(data[['Date', 'Signal']])

st.subheader("News Feed")
news = fetch_news(ticker)
for article in news:
    st.write(article)

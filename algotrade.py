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

# Algorithmic Trading Strategies
def mean_reversion_strategy(data):
    data['Position'] = np.where(data['RSI'] < 30, 1, 0)
    data['Position'] = np.where(data['RSI'] > 70, -1, data['Position'])
    data['Signal'] = data['Position'].diff()
    return data

def moving_average_strategy(data):
    data['Position'] = np.where(data['SMA_20'] > data['EMA_20'], 1, -1)
    data['Signal'] = data['Position'].diff()
    return data

def volume_weighted_average_price(data):
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['Position'] = np.where(data['Close'] > data['VWAP'], 1, -1)
    data['Signal'] = data['Position'].diff()
    return data

def bollinger_band_strategy(data):
    data['Position'] = np.where(data['Close'] < data['BB_Low'], 1, 0)
    data['Position'] = np.where(data['Close'] > data['BB_High'], -1, data['Position'])
    data['Signal'] = data['Position'].diff()
    return data

def macd_strategy(data):
    data['Position'] = np.where(data['MACD'] > data['MACD_Signal'], 1, -1)
    data['Signal'] = data['Position'].diff()
    return data

# Apply strategies
data = mean_reversion_strategy(data)
data = moving_average_strategy(data)
data = volume_weighted_average_price(data)
data = bollinger_band_strategy(data)
data = macd_strategy(data)

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

# Fetch S&P 500 Constituents
def fetch_constituents():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(str(table))[0]
    return df

# Fetch Related Indices
def fetch_related_indices():
    url = 'https://finance.yahoo.com/quote/%5EGSPC/components/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    related_indices = []
    for link in soup.find_all('a', {'data-test': 'quoteLink'}):
        related_indices.append(link.text)
    return related_indices

# Fetch Market Sentiments
def fetch_market_sentiments():
    url = 'https://www.investing.com/indices/us-spx-500'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    sentiment = soup.find('div', {'class': 'sentiment'}).text.strip()
    return sentiment

# Display the data and plots
st.subheader("Price Data")
st.write(data)

st.subheader("Candlestick Chart with Technical Indicators")
st.plotly_chart(plot_data(data))

# Header Information
st.subheader("Header Information")
st.write(f"Current Index Value: {data['Close'].iloc[-1]}")
st.write(f"Last Update Time: {datetime.now()}")

# Overview Section
st.subheader("Overview Section")
st.write("Basic description of the S&P 500 index")

# Historical Data
st.subheader("Historical Data")
st.write(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])

# Constituents
st.subheader("Constituents")
constituents = fetch_constituents()
st.write(constituents)

# Related Indices
st.subheader("Related Indices")
related_indices = fetch_related_indices()
st.write(related_indices)

# Market Sentiments
st.subheader("Market Sentiments")
sentiments = fetch_market_sentiments()
st.write(sentiments)

# Technical Summary
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

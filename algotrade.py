import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import requests
from bs4 import BeautifulSoup

# Set the title and layout of the Streamlit app
st.set_page_config(layout="wide")
st.title("Algorithm-Driven Trading App for US Indices")

# Sidebar for user inputs
st.sidebar.title("User Inputs")
indices = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC"}
index_name = st.sidebar.selectbox("Select Index", list(indices.keys()))
ticker = indices[index_name]
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
timeframe = st.sidebar.selectbox("Timeframe", ["1 Week", "1 Month", "1 Year", "2 Years", "3 Years", "4 Years", "5 Years"])

# Adjust date range based on timeframe
if timeframe == "1 Week":
    start_date = datetime.today() - timedelta(days=7)
elif timeframe == "1 Month":
    start_date = datetime.today() - timedelta(days=30)
elif timeframe == "1 Year":
    start_date = datetime.today() - timedelta(days=365)
elif timeframe == "2 Years":
    start_date = datetime.today() - timedelta(days=730)
elif timeframe == "3 Years":
    start_date = datetime.today() - timedelta(days=1095)
elif timeframe == "4 Years":
    start_date = datetime.today() - timedelta(days=1460)
elif timeframe == "5 Years":
    start_date = datetime.today() - timedelta(days=1825)

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

# General market sentiment (example: based on RSI)
def market_sentiment(data):
    last_rsi = data["RSI"].iloc[-1]
    if last_rsi > 70:
        return "Bearish"
    elif last_rsi < 30:
        return "Bullish"
    else:
        return "Neutral"

sentiment = market_sentiment(data)

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
    fig.update_layout(title="Price with Technical Indicators",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      xaxis_rangeslider_visible=False)
    
    return fig

# Display the data and plots
st.subheader(f"{index_name} Data")
st.write(f"General Market Sentiment: **{sentiment}**")

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

# Technical Chart Info
st.subheader("Technical Chart Info")
info = {
    "Day's Range": f"{data['Low'].min():.2f} - {data['High'].max():.2f}",
    "52 wk Range": f"{data['Low'].min():.2f} - {data['High'].max():.2f}",
    "Prev. Close": f"{data['Close'].iloc[-2]:.2f}",
    "Open": f"{data['Open'].iloc[-1]:.2f}",
    "1-Year Change": f"{((data['Close'].iloc[-1] - data['Close'].iloc[-252])/data['Close'].iloc[-252])*100:.2f}%",
    "Type": "Indices",
    "Market": "United States",
    "Number Of Components": "502",
    "Volume": "-",
    "Average Vol.(3m)": "2,362,808,064"
}

for key, value in info.items():
    st.write(f"**{key}:** {value}")

st.subheader("News Feed")
news = fetch_news(ticker)
for article in news:
    st.write(article)

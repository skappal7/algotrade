import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import plotly.express as px

# Set Streamlit configuration
st.set_page_config(
    page_title="S&P 500 Technical Analysis and Predictions",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define sidebar inputs
st.sidebar.title("Settings")
ticker = st.sidebar.selectbox("Select Ticker:", options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB'])
start_date = st.sidebar.date_input("Start Date", value=datetime.date.today() - datetime.timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())
zoom_in = st.sidebar.button("Zoom In")
zoom_out = st.sidebar.button("Zoom Out")

# Fetch stock data
@st.cache
def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

data = get_stock_data(ticker, start_date, end_date)

# Zoom functionality
if zoom_in:
    start_date += datetime.timedelta(days=10)
    end_date -= datetime.timedelta(days=10)
if zoom_out:
    start_date -= datetime.timedelta(days=10)
    end_date += datetime.timedelta(days=10)
    data = get_stock_data(ticker, start_date, end_date)

# Market Sentiment
close_price = data['Close'][-1]
ma_50 = SMAIndicator(close=data['Close'], window=50).sma_indicator()[-1]
rsi = RSIIndicator(close=data['Close'], window=14).rsi()[-1]

sentiment = "Bullish" if close_price > ma_50 and rsi > 50 else "Bearish" if close_price < ma_50 and rsi < 50 else "Neutral"
st.markdown(f"### Market Sentiment: {sentiment}")

# Plot candlestick chart with Bollinger Bands
fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
bollinger_bands = BollingerBands(close=data['Close'])
data['bb_h'] = bollinger_bands.bollinger_hband()
data['bb_l'] = bollinger_bands.bollinger_lband()
fig.add_trace(go.Scatter(x=data.index, y=data['bb_h'], name='Bollinger High', line=dict(color='rgba(255, 0, 0, 0.2)')))
fig.add_trace(go.Scatter(x=data.index, y=data['bb_l'], name='Bollinger Low', line=dict(color='rgba(0, 255, 0, 0.2)')))
fig.update_layout(title=f'{ticker} Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig)

# Moving Averages Table
ma_periods = [5, 10, 20, 50, 100, 200]
ma_table = pd.DataFrame(index=ma_periods, columns=['Simple MA', 'Simple Recommendation', 'Exponential MA', 'Exponential Recommendation'])
for period in ma_periods:
    sma = SMAIndicator(close=data['Close'], window=period).sma_indicator()[-1]
    ema = EMAIndicator(close=data['Close'], window=period).ema_indicator()[-1]
    sma_recommendation = "Buy" if close_price > sma else "Sell"
    ema_recommendation = "Buy" if close_price > ema else "Sell"
    ma_table.loc[period] = [round(sma, 3), sma_recommendation, round(ema, 3), ema_recommendation]

st.markdown("### Moving Averages")
st.dataframe(ma_table)

# Momentum Oscillators Table
rsi = RSIIndicator(close=data['Close'], window=14).rsi()[-1]
stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14).stoch()[-1]
stoch_rsi = StochRSIIndicator(close=data['Close'], window=14).stochrsi()[-1]
williams_r = (data['Close'] - data['Low']) / (data['High'] - data['Low']) * -100
cci = (data['Close'] - data['Close'].rolling(window=20).mean()) / (0.015 * data['Close'].rolling(window=20).std())
roc = (data['Close'] - data['Close'].shift(12)) / data['Close'].shift(12) * 100
ultimate_oscillator = (4 * data['Close'].rolling(window=7).mean() + 2 * data['Close'].rolling(window=14).mean() + data['Close'].rolling(window=28).mean()) / 7

oscillators = pd.DataFrame({
    'Name': ['RSI(14)', 'STOCH(9,6)', 'STOCHRSI(14)', 'Williams %R', 'CCI(14)', 'ROC', 'Ultimate Oscillator'],
    'Value': [rsi, stoch, stoch_rsi, williams_r, cci[-1], roc[-1], ultimate_oscillator[-1]],
    'Action': [
        "Sell" if rsi > 70 else "Buy" if rsi < 30 else "",
        "Sell" if stoch > 80 else "Buy" if stoch < 20 else "",
        "Sell" if stoch_rsi > 70 else "Buy" if stoch_rsi < 30 else "",
        "Sell" if williams_r > -20 else "Buy" if williams_r < -80 else "",
        "Sell" if cci[-1] > 100 else "Buy" if cci[-1] < -100 else "",
        "Buy" if roc[-1] > 0 else "Sell",
        "Sell" if ultimate_oscillator[-1] > 70 else "Buy" if ultimate_oscillator[-1] < 30 else ""
    ]
})
st.markdown("### Momentum Oscillators")
st.dataframe(oscillators)

# Volatility Table
atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()[-1]
highs_lows = data['High'] - data['Low']
volatility = pd.DataFrame({
    'Name': ['ATR(14)', 'Highs/Lows(14)'],
    'Value': [atr, highs_lows.mean()],
    'Action': ['High Volatility' if atr > 10 else 'Less Volatility', 'Buy' if highs_lows.mean() > 0 else 'Sell']
})
st.markdown("### Volatility")
st.dataframe(volatility)

# Pivot Points Table
pivot = (data['High'][-1] + data['Low'][-1] + data['Close'][-1]) / 3
s1 = (2 * pivot) - data['High'][-1]
s2 = pivot - (data['High'][-1] - data['Low'][-1])
s3 = s1 - (data['High'][-1] - data['Low'][-1])
r1 = (2 * pivot) - data['Low'][-1]
r2 = pivot + (data['High'][-1] - data['Low'][-1])
r3 = r1 + (data['High'][-1] - data['Low'][-1])

pivot_points = pd.DataFrame({
    'Name': ['Classic'],
    'S3': [s3],
    'S2': [s2],
    'S1': [s1],
    'PP': [pivot],
    'R1': [r1],
    'R2': [r2],
    'R3': [r3]
})
st.markdown("### Pivot Points")
st.dataframe(pivot_points)

# Historical Data Table
st.markdown("### Historical Data")
freq = st.radio("Select Frequency:", ('Daily', 'Weekly', 'Monthly'))
if freq == 'Weekly':
    data = data.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
elif freq == 'Monthly':
    data = data.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
data['Change'] = data['Close'] - data['Close'].shift(1)
data['ChangePct'] = data['Change'] / data['Close'].shift(1) * 100
st.dataframe(data[['Close', 'Open', 'High', 'Low', 'Volume', 'Change', 'ChangePct']].dropna())

# Gainers and Losers Table
st.markdown("### Gainers and Losers")
gainers_losers_date_range = st.date_input("Select Date Range:", [datetime.date.today() - datetime.timedelta(days=30), datetime.date.today()])
gainers = data[data.index >= gainers_losers_date_range[0]].nlargest(5, 'ChangePct')
losers = data[data.index >= gainers_losers_date_range[0]].nsmallest(5, 'ChangePct')

st.markdown("#### Most Active Gainers")
st.dataframe(gainers[['Close', 'Change', 'ChangePct']])
st.markdown("#### Most Active Losers")
st.dataframe(losers[['Close', 'Change', 'ChangePct']])

# Key Definitions
st.markdown("""
### Key Definitions of Metrics
#### Relative Strength Index (RSI)
RSI is a momentum oscillator that measures the speed and change of price movements. RSI values range from 0 to 100. Traditionally, RSI is considered overbought when above 70 and oversold when below 30.

#### Stochastic Oscillator (STOCH)
The Stochastic Oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. Sensitivity to market movements can be reduced by adjusting the time period or by taking a moving average of the result.

#### Stochastic RSI (STOCHRSI)
STOCHRSI applies the Stochastic Oscillator formula to RSI values, instead of price data. It is more sensitive to recent price changes than RSI.

#### Williams %R
Williams %R is a momentum indicator that measures overbought and oversold levels, similar to the Stochastic Oscillator. It ranges from 0 to -100.

#### Commodity Channel Index (CCI)
CCI measures the deviation of the price from its average price over a period of time. High positive readings indicate the price is well above its average, while low negative readings indicate the price is well below its average.

#### Rate of Change (ROC)
ROC is a momentum oscillator that measures the percentage change in price between the current price and the price n periods ago.

#### Ultimate Oscillator
The Ultimate Oscillator is a momentum oscillator designed to capture momentum across three different timeframes.

#### Average True Range (ATR)
ATR measures market volatility by decomposing the entire range of an asset price for that period. It is used to measure volatility, with higher values indicating higher volatility.

#### Highs/Lows
This indicator calculates the difference between the highest and lowest prices over a specific period. It is used to identify breakout and breakdown levels.

#### Moving Average Convergence Divergence (MACD)
MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period EMA from the 12-period EMA.

#### Average Directional Index (ADX)
ADX measures the strength of a trend. Readings above 20 indicate a strong trend, while readings below 20 indicate a weak trend.

#### Bull/Bear Power
Bull/Bear Power measures the buying or selling pressure in the market. Bull Power is calculated by subtracting the 13-period EMA from the high of the day. Bear Power is calculated by subtracting the 13-period EMA from the low of the day.

#### Pivot Points
Pivot points are used to identify potential support and resistance levels. They are calculated based on the high, low, and closing prices of the previous day.

##### Classic
Classic pivot points are calculated using the standard formula: PP = (High + Low + Close) / 3, with support and resistance levels calculated based on this pivot point.

##### Fibonacci
Fibonacci pivot points use Fibonacci retracement levels to calculate support and resistance levels.

##### Camarilla
Camarilla pivot points use the closing price and a scaling factor to calculate support and resistance levels.

##### Woodie's
Woodie's pivot points give more weight to the closing price.

##### DeMark's
DeMark's pivot points are calculated differently depending on whether the close is higher, lower, or equal to the open.
""")

import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="S&P 500 Technical Analysis", layout="wide")
st.sidebar.title("S&P 500 Technical Analysis")

# Function to get stock data
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

# Sidebar inputs
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))

tabs = st.tabs(["US Indices", "Key Definitions"])

with tabs[0]:
    st.sidebar.title("US Indices Settings")
    ticker = st.sidebar.selectbox("Select Ticker:", options=["^GSPC", "^DJI", "^IXIC", "^RUT"])
    data = get_stock_data(ticker, start_date, end_date)

    # Market Sentiment
    close_price = data['Close'].iloc[-1]
    ma_50 = SMAIndicator(close=data['Close'], window=50).sma_indicator().iloc[-1]
    rsi = RSIIndicator(close=data['Close'], window=14).rsi().iloc[-1]

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
        sma = SMAIndicator(close=data['Close'], window=period).sma_indicator().iloc[-1]
        ema = EMAIndicator(close=data['Close'], window=period).ema_indicator().iloc[-1]
        sma_recommendation = "Buy" if close_price > sma else "Sell"
        ema_recommendation = "Buy" if close_price > ema else "Sell"
        ma_table.loc[period] = [round(sma, 3), sma_recommendation, round(ema, 3), ema_recommendation]

    st.markdown("### Moving Averages")
    st.dataframe(ma_table)

    # Momentum Oscillators Table
    rsi = RSIIndicator(close=data['Close'], window=14).rsi().iloc[-1]
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14).stoch().iloc[-1]
    stoch_rsi = StochRSIIndicator(close=data['Close'], window=14).stochrsi().iloc[-1]
    williams_r = ((data['Close'] - data['Low']) / (data['High'] - data['Low']) * -100).iloc[-1]
    cci = ((data['Close'] - data['Close'].rolling(window=20).mean()) / (0.015 * data['Close'].rolling(window=20).std())).iloc[-1]
    roc = ((data['Close'] - data['Close'].shift(12)) / data['Close'].shift(12) * 100).iloc[-1]
    ultimate_oscillator = ((4 * data['Close'].rolling(window=7).mean() + 2 * data['Close'].rolling(window=14).mean() + data['Close'].rolling(window=28).mean()) / 7).iloc[-1]

    oscillators = pd.DataFrame({
        'Name': ['RSI(14)', 'STOCH(9,6)', 'STOCHRSI(14)', 'Williams %R', 'CCI(14)', 'ROC', 'Ultimate Oscillator'],
        'Value': [rsi, stoch, stoch_rsi, williams_r, cci, roc, ultimate_oscillator],
        'Action': [
            "Sell" if rsi > 70 else "Buy" if rsi < 30 else "",
            "Sell" if stoch > 80 else "Buy" if stoch < 20 else "",
            "Sell" if stoch_rsi > 70 else "Buy" if stoch_rsi < 30 else "",
            "Sell" if williams_r > -20 else "Buy" if williams_r < -80 else "",
            "Sell" if cci > 100 else "Buy" if cci < -100 else "",
            "Buy" if roc > 0 else "Sell",
            "Sell" if ultimate_oscillator > 70 else "Buy" if ultimate_oscillator < 30 else ""
        ]
    })
    st.markdown("### Momentum Oscillators")
    st.dataframe(oscillators)

    # Volatility Table
    atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range().iloc[-1]
    highs_lows = (data['High'] - data['Low']).mean()
    volatility = pd.DataFrame({
        'Name': ['ATR(14)', 'Highs/Lows(14)'],
        'Value': [atr, highs_lows],
        'Action': ['High Volatility' if atr > 10 else 'Less Volatility', 'Buy' if highs_lows > 0 else 'Sell']
    })
    st.markdown("### Volatility")
    st.dataframe(volatility)

    # Pivot Points Table
    high = data['High'].iloc[-1]
    low = data['Low'].iloc[-1]
    close = data['Close'].iloc[-1]
    
    # Classic Pivot Points
    pivot = (high + low + close) / 3
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = s1 - (high - low)
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = r1 + (high - low)
    
    # Fibonacci Pivot Points
    fib_r1 = pivot + (high - low) * 0.382
    fib_r2 = pivot + (high - low) * 0.618
    fib_r3 = pivot + (high - low) * 1.000
    fib_s1 = pivot - (high - low) * 0.382
    fib_s2 = pivot - (high - low) * 0.618
    fib_s3 = pivot - (high - low) * 1.000
    
    # Camarilla Pivot Points
    camarilla_r4 = close + 1.1 * (high - low) / 2
    camarilla_r3 = close + 1.1 * (high - low) / 4
    camarilla_r2 = close + 1.1 * (high - low) / 6
    camarilla_r1 = close + 1.1 * (high - low) / 12
    camarilla_s1 = close - 1.1 * (high - low) / 12
    camarilla_s2 = close - 1.1 * (high - low) / 6
    camarilla_s3 = close - 1.1 * (high - low) / 4
    camarilla_s4 = close - 1.1 * (high - low) / 2
    
    # Woodie's Pivot Points
    woodie_pivot = (high + low + 2 * close) / 4
    woodie_r1 = (2 * woodie_pivot) - low
    woodie_r2 = woodie_pivot + (high - low)
    woodie_r3 = woodie_r1 + (high - low)
    woodie_s1 = (2 * woodie_pivot) - high
    woodie_s2 = woodie_pivot - (high - low)
    woodie_s3 = woodie_s1 - (high - low)
    
    # DeMark's Pivot Points
    demark_pp = (high + low + close) / 3
    demark_r1 = demark_pp + (demark_pp - low)
    demark_s1 = demark_pp - (high - demark_pp)
    
    pivot_points = pd.DataFrame({
        'Name': ['Classic', 'Fibonacci', 'Camarilla', "Woodie's", "DeMark's"],
        'S3': [s3, fib_s3, camarilla_s4, woodie_s3, None],
        'S2': [s2, fib_s2, camarilla_s3, woodie_s2, None],
        'S1': [s1, fib_s1, camarilla_s2, woodie_s1, demark_s1],
        'PP': [pivot, pivot, pivot, woodie_pivot, demark_pp],
        'R1': [r1, fib_r1, camarilla_r1, woodie_r1, demark_r1],
        'R2': [r2, fib_r2, camarilla_r2, woodie_r2, None],
        'R3': [r3, fib_r3, camarilla_r3, woodie_r3, None],
        'R4': [None, None, camarilla_r4, None, None]
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

with tabs[1]:
    st.markdown("### Key Definitions of Metrics")
    st.markdown("""
    **Relative Strength Index (RSI)**: RSI is a momentum oscillator that measures the speed and change of price movements. RSI values range from 0 to 100. Traditionally, RSI is considered overbought when above 70 and oversold when below 30.

    **Stochastic Oscillator (STOCH)**: The Stochastic Oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. Sensitivity to market movements can be reduced by adjusting the time period or by taking a moving average of the result.

    **Stochastic RSI (STOCHRSI)**: STOCHRSI applies the Stochastic Oscillator formula to RSI values, instead of price data. It is more sensitive to recent price changes than RSI.

    **Williams %R**: Williams %R is a momentum indicator that measures overbought and oversold levels, similar to the Stochastic Oscillator. It ranges from 0 to -100.

    **Commodity Channel Index (CCI)**: CCI measures the deviation of the price from its average price over a period of time. High positive readings indicate the price is well above its average, while low negative readings indicate the price is well below its average.

    **Rate of Change (ROC)**: ROC is a momentum oscillator that measures the percentage change in price between the current price and the price n periods ago.

    **Ultimate Oscillator**: The Ultimate Oscillator is a momentum oscillator designed to capture momentum across three different timeframes.

    **Average True Range (ATR)**: ATR measures market volatility by decomposing the entire range of an asset price for that period. It is used to measure volatility, with higher values indicating higher volatility.

    **Highs/Lows**: This indicator calculates the difference between the highest and lowest prices over a specific period. It is used to identify breakout and breakdown levels.

    **Moving Average Convergence Divergence (MACD)**: MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period EMA from the 12-period EMA.

    **Average Directional Index (ADX)**: ADX measures the strength of a trend. Readings above 20 indicate a strong trend, while readings below 20 indicate a weak trend.

    **Bull/Bear Power**: Bull/Bear Power measures the buying or selling pressure in the market. Bull Power is calculated by subtracting the 13-period EMA from the high of the day. Bear Power is calculated by subtracting the 13-period EMA from the low of the day.

    **Pivot Points**: Pivot points are used to identify potential support and resistance levels. They are calculated based on the high, low, and closing prices of the previous day.

    **Classic**: Classic pivot points are calculated using the standard formula: PP = (High + Low + Close) / 3, with support and resistance levels calculated based on this pivot point.

    **Fibonacci**: Fibonacci pivot points use Fibonacci retracement levels to calculate support and resistance levels.

    **Camarilla**: Camarilla pivot points use the closing price and a scaling factor to calculate support and resistance levels.

    **Woodie's**: Woodie's pivot points give more weight to the closing price.

    **DeMark's**: DeMark's pivot points are calculated differently depending on whether the close is higher, lower, or equal to the open.
    """)

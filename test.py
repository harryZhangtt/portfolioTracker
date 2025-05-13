import yfinance as yf

ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
print(data.head())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf

# download data online
ticker = 'MSFT'
downloaded = yf.download(ticker, start = '2005-1-1', end = '2011-1-1', interval= '1d')
returns = 100 * downloaded['Close'].pct_change().dropna()

# multifigures

fig = plt.figure('garch model forecasting')

obj1 = fig.add_subplot(221)
plt.plot(returns, label='prices')
plt.ylabel('returns')
plt.title('volatility')
plt.legend()

# rolling GARCH

# first check the pacf
# plot_pacf(returns**2)
# plt.show()
# plt.pause(3)
# plt.close()

empty_list = []
rolling_number = 250



for i in range(rolling_number):
    train_set = returns[:-(rolling_number-i)]
    model = arch_model(train_set, p=2, q=0)
    model_fit = model.fit(disp='off')
    prediction = model_fit.forecast(horizon=1)
    empty_list.append(prediction.variance.values[-1][0])

empty_list = pd.Series(data=empty_list,index=returns.index[-250:])

obj2 = fig.add_subplot(222)
plt.plot(empty_list)
plt.title('rolling forecasting')
plt.ylabel('one day predictions')


obj3 = fig.add_subplot(212)
plt.plot(empty_list, color= 'orange', linestyle= '--', label='one day predictions')
plt.plot(returns[-250:]**2, label= 'time series-train')
plt.legend()

plt.show()






















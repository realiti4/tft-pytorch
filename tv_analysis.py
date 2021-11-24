import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tradingfeatures import binance

binance = binance()

df = pd.read_csv('ideas_trend.csv', index_col='timestamp')


# Merge with prices
prices = binance.get(5000)

columns = ['longs', 'shorts', 'close']
merged = df.join(prices)
merged = merged[columns]


# pyplot twin axis solution here



# normalize
array = merged.to_numpy()
array, prices = array[:, :2], array[:, 2]

# sums = array.sum(1)[:, None]
# norm = array / sums

# longs = norm[:, 0]

# plot
ax = plt.subplot(211)
ax2 = plt.subplot(212)
# ax.set_ylim(0, 1)

ax.plot(array)
ax2.plot(prices)


plt.show()



print('de')
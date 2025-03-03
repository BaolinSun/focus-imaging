import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

df = pd.read_csv(r'rfdata\rfdata_1_32.csv', sep=',', header=None)

# df = df.iloc[200:]

data = df.values
data = (data - 512) / 512

print(data.shape)

# plt.plot(df[0])
# plt.plot(df[31])
# plt.plot(df[32])
# plt.plot(df[63])
for i in range(64):
    plt.plot(df[i])
plt.grid(True)
plt.ylim([0, 1024])
plt.show()
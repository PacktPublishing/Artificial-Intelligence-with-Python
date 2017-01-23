import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from timeseries import read_data

# Load input data
index = 2
data = read_data('data_2D.txt', index)

# Plot data with year-level granularity 
start = '2003'
end = '2011'
plt.figure()
data[start:end].plot()
plt.title('Input data from ' + start + ' to ' + end)

# Plot data with month-level granularity 
start = '1998-2'
end = '2006-7'
plt.figure()
data[start:end].plot()
plt.title('Input data from ' + start + ' to ' + end)

plt.show()
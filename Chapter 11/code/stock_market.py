import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo_ochl\
        as quotes_yahoo
from hmmlearn.hmm import GaussianHMM

# Load historical stock quotes from matplotlib package 
start = datetime.date(1970, 9, 4) 
end = datetime.date(2016, 5, 17)
stock_quotes = quotes_yahoo('INTC', start, end) 

# Extract the closing quotes everyday
closing_quotes = np.array([quote[2] for quote in stock_quotes])

# Extract the volume of shares traded everyday 
volumes = np.array([quote[5] for quote in stock_quotes])[1:]

# Take the percentage difference of closing stock prices
diff_percentages = 100.0 * np.diff(closing_quotes) / closing_quotes[:-1]

# Take the list of dates starting from the second value
dates = np.array([quote[0] for quote in stock_quotes], dtype=np.int)[1:]

# Stack the differences and volume values column-wise for training
training_data = np.column_stack([diff_percentages, volumes])

# Create and train Gaussian HMM 
hmm = GaussianHMM(n_components=7, covariance_type='diag', n_iter=1000)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    hmm.fit(training_data)

# Generate data using the HMM model
num_samples = 300 
samples, _ = hmm.sample(num_samples) 

# Plot the difference percentages 
plt.figure()
plt.title('Difference percentages')
plt.plot(np.arange(num_samples), samples[:, 0], c='black')

# Plot the volume of shares traded
plt.figure()
plt.title('Volume of shares')
plt.plot(np.arange(num_samples), samples[:, 1], c='black')
plt.ylim(ymin=0)

plt.show()


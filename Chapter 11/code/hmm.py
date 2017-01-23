import datetime

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from timeseries import read_data 

# Load input data
data = np.loadtxt('data_1D.txt', delimiter=',')

# Extract the data column (third column) for training 
X = np.column_stack([data[:, 2]])

# Create a Gaussian HMM 
num_components = 5
hmm = GaussianHMM(n_components=num_components, 
        covariance_type='diag', n_iter=1000)

# Train the HMM 
print('\nTraining the Hidden Markov Model...')
hmm.fit(X)

# Print HMM stats
print('\nMeans and variances:')
for i in range(hmm.n_components):
    print('\nHidden state', i+1)
    print('Mean =', round(hmm.means_[i][0], 2))
    print('Variance =', round(np.diag(hmm.covars_[i])[0], 2))

# Generate data using the HMM model
num_samples = 1200
generated_data, _ = hmm.sample(num_samples) 
plt.plot(np.arange(num_samples), generated_data[:, 0], c='black')
plt.title('Generated data')

plt.show()
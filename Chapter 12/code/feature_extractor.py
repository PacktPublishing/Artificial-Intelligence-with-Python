import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile 
from features import mfcc, logfbank

# Read the input audio file
sampling_freq, signal = wavfile.read('random_sound.wav')

# Take the first 10,000 samples for analysis
signal = signal[:10000]

# Extract the MFCC features 
features_mfcc = mfcc(signal, sampling_freq)

# Print the parameters for MFCC
print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])

# Plot the features
features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

# Extract the Filter Bank features
features_fb = logfbank(signal, sampling_freq)

# Print the parameters for Filter Bank 
print('\nFilter bank:\nNumber of windows =', features_fb.shape[0])
print('Length of each feature =', features_fb.shape[1])

# Plot the features
features_fb = features_fb.T
plt.matshow(features_fb)
plt.title('Filter bank')

plt.show()

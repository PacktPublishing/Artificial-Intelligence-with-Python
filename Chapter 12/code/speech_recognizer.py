import os
import argparse 
import warnings

import numpy as np
from scipy.io import wavfile 

from hmmlearn import hmm
from features import mfcc

# Define a function to parse the input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM-based speech \
            recognition system')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Input folder containing the audio files for training")
    return parser

# Define a class to train the HMM 
class ModelHMM(object):
    def __init__(self, num_components=4, num_iter=1000):
        self.n_components = num_components
        self.n_iter = num_iter

        self.cov_type = 'diag' 
        self.model_name = 'GaussianHMM' 

        self.models = []

        self.model = hmm.GaussianHMM(n_components=self.n_components, 
                covariance_type=self.cov_type, n_iter=self.n_iter)

    # 'training_data' is a 2D numpy array where each row is 13-dimensional
    def train(self, training_data):
        np.seterr(all='ignore')
        cur_model = self.model.fit(training_data)
        self.models.append(cur_model)

    # Run the HMM model for inference on input data
    def compute_score(self, input_data):
        return self.model.score(input_data)

# Define a function to build a model for each word
def build_models(input_folder):
    # Initialize the variable to store all the models
    speech_models = []

    # Parse the input directory
    for dirname in os.listdir(input_folder):
        # Get the name of the subfolder 
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder): 
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]

        # Initialize the variables
        X = np.array([])

        # Create a list of files to be used for training
        # We will leave one file per folder for testing
        training_files = [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]

        # Iterate through the training files and build the models
        for filename in training_files: 
            # Extract the current filepath
            filepath = os.path.join(subfolder, filename)

            # Read the audio signal from the input file
            sampling_freq, signal = wavfile.read(filepath)
            
            # Extract the MFCC features
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features_mfcc = mfcc(signal, sampling_freq)

            # Append to the variable X
            if len(X) == 0:
                X = features_mfcc
            else:
                X = np.append(X, features_mfcc, axis=0)
            
        # Create the HMM model
        model = ModelHMM()

        # Train the HMM
        model.train(X)

        # Save the model for the current word
        speech_models.append((model, label))

        # Reset the variable
        model = None

    return speech_models

# Define a function to run tests on input files
def run_tests(test_files):
    # Classify input data
    for test_file in test_files:
        # Read input file
        sampling_freq, signal = wavfile.read(test_file)

        # Extract MFCC features
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(signal, sampling_freq)

        # Define variables
        max_score = -float('inf') 
        output_label = None 

        # Run the current feature vector through all the HMM
        # models and pick the one with the highest score
        for item in speech_models:
            model, label = item
            score = model.compute_score(features_mfcc)
            if score > max_score:
                max_score = score
                predicted_label = label

        # Print the predicted output 
        start_index = test_file.find('/') + 1
        end_index = test_file.rfind('/')
        original_label = test_file[start_index:end_index]
        print('\nOriginal: ', original_label) 
        print('Predicted:', predicted_label)

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder

    # Build an HMM model for each word
    speech_models = build_models(input_folder)

    # Test files -- the 15th file in each subfolder 
    test_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if '15' in x):
            filepath = os.path.join(root, filename)
            test_files.append(filepath)

    run_tests(test_files)
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Synthesize the tone based on the input parameters
def tone_synthesizer(freq, duration, amplitude=1.0, sampling_freq=44100):
    # Construct the time axis 
    time_axis = np.linspace(0, duration, duration * sampling_freq)

    # Construct the audio signal
    signal = amplitude * np.sin(2 * np.pi * freq * time_axis)

    return signal.astype(np.int16) 

if __name__=='__main__':
    # Names of output files
    file_tone_single = 'generated_tone_single.wav'
    file_tone_sequence = 'generated_tone_sequence.wav'

    # Source: http://www.phy.mtu.edu/~suits/notefreqs.html
    mapping_file = 'tone_mapping.json'
    
    # Load the tone to frequency map from the mapping file
    with open(mapping_file, 'r') as f:
        tone_map = json.loads(f.read())
        
    # Set input parameters to generate 'F' tone
    tone_name = 'F'
    duration = 3     # seconds
    amplitude = 12000
    sampling_freq = 44100    # Hz

    # Extract the tone frequency
    tone_freq = tone_map[tone_name]

    # Generate the tone using the above parameters
    synthesized_tone = tone_synthesizer(tone_freq, duration, amplitude, sampling_freq)

    # Write the audio signal to the output file
    write(file_tone_single, sampling_freq, synthesized_tone)

    # Define the tone sequence along with corresponding durations in seconds
    tone_sequence = [('G', 0.4), ('D', 0.5), ('F', 0.3), ('C', 0.6), ('A', 0.4)]

    # Construct the audio signal based on the above sequence 
    signal = np.array([])
    for item in tone_sequence:
        # Get the name of the tone 
        tone_name = item[0]

        # Extract the corresponding frequency of the tone
        freq = tone_map[tone_name]

        # Extract the duration
        duration = item[1]

        # Synthesize the tone
        synthesized_tone = tone_synthesizer(freq, duration, amplitude, sampling_freq)

        # Append the output signal
        signal = np.append(signal, synthesized_tone, axis=0)

    # Save the audio in the output file
    write(file_tone_sequence, sampling_freq, signal)
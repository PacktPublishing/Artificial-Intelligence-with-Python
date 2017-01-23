import os
import sys

import cv2
import numpy as np

# Define the input file 
input_file = 'letter.data' 

# Define the visualization parameters 
img_resize_factor = 12
start = 6
end = -1
height, width = 16, 8

# Iterate until the user presses the Esc key
with open(input_file, 'r') as f:
    for line in f.readlines():
        # Read the data
        data = np.array([255 * float(x) for x in line.split('\t')[start:end]])

        # Reshape the data into a 2D image
        img = np.reshape(data, (height, width))

        # Scale the image
        img_scaled = cv2.resize(img, None, fx=img_resize_factor, fy=img_resize_factor)

        # Display the image
        cv2.imshow('Image', img_scaled)

        # Check if the user pressed the Esc key
        c = cv2.waitKey()
        if c == 27:
            break

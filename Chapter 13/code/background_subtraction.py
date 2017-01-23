import cv2
import numpy as np

# Define a function to get the current frame from the webcam
def get_frame(cap, scaling_factor):
    # Read the current frame from the video capture object
    _, frame = cap.read()

    # Resize the image
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame

if __name__=='__main__':
    # Define the video capture object
    cap = cv2.VideoCapture(0)

    # Define the background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
     
    # Define the number of previous frames to use to learn. 
    # This factor controls the learning rate of the algorithm. 
    # The learning rate refers to the rate at which your model 
    # will learn about the background. Higher value for 
    # ‘history’ indicates a slower learning rate. You can 
    # play with this parameter to see how it affects the output.
    history = 100

    # Define the learning rate
    learning_rate = 1.0/history

    # Keep reading the frames from the webcam 
    # until the user hits the 'Esc' key
    while True:
        # Grab the current frame
        frame = get_frame(cap, 0.5)

        # Compute the mask 
        mask = bg_subtractor.apply(frame, learningRate=learning_rate)

        # Convert grayscale image to RGB color image
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Display the images
        cv2.imshow('Input', frame)
        cv2.imshow('Output', mask & frame)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(10)
        if c == 27:
            break

    # Release the video capture object
    cap.release()
    
    # Close all the windows
    cv2.destroyAllWindows()

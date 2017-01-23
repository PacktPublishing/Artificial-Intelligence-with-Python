import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the number of points to generate
num_points = 1200

# Generate the data based on equation y = mx + c
data = []
m = 0.2
c = 0.5
for i in range(num_points):
    # Generate 'x' 
    x = np.random.normal(0.0, 0.8)

    # Generate some noise
    noise = np.random.normal(0.0, 0.04)

    # Compute 'y' 
    y = m*x + c + noise 

    data.append([x, y])

# Separate x and y
x_data = [d[0] for d in data]
y_data = [d[1] for d in data]

# Plot the generated data
plt.plot(x_data, y_data, 'ro')
plt.title('Input data')
plt.show()

# Generate weights and biases
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# Define equation for 'y'
y = W * x_data + b

# Define how to compute the loss
loss = tf.reduce_mean(tf.square(y - y_data))

# Define the gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Initialize all the variables
init = tf.initialize_all_variables()

# Start the tensorflow session and run it
sess = tf.Session()
sess.run(init)

# Start iterating
num_iterations = 10
for step in range(num_iterations):
    # Run the session
    sess.run(train)

    # Print the progress
    print('\nITERATION', step+1)
    print('W =', sess.run(W)[0])
    print('b =', sess.run(b)[0])
    print('loss =', sess.run(loss))

    # Plot the input data 
    plt.plot(x_data, y_data, 'ro')

    # Plot the predicted output line
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))

    # Set plotting parameters
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title('Iteration ' + str(step+1) + ' of ' + str(num_iterations))
    plt.show()


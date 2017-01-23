import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Build a classifier using \
            MNIST data')
    parser.add_argument('--input-dir', dest='input_dir', type=str, 
            default='./mnist_data', help='Directory for storing data')
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    # Get the MNIST data
    mnist = input_data.read_data_sets(args.input_dir, one_hot=True)

    # The images are 28x28, so create the input layer 
    # with 784 neurons (28x28=784) 
    x = tf.placeholder(tf.float32, [None, 784])

    # Create a layer with weights and biases. There are 10 distinct 
    # digits, so the output layer should have 10 classes
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Create the equation for 'y' using y = W*x + b
    y = tf.matmul(x, W) + b

    # Define the entropy loss and the gradient descent optimizer
    y_loss = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_loss))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Initialize all the variables 
    init = tf.initialize_all_variables()

    # Create a session
    session = tf.Session()
    session.run(init)

    # Start training
    num_iterations = 1200
    batch_size = 90
    for _ in range(num_iterations):
        # Get the next batch of images
        x_batch, y_batch = mnist.train.next_batch(batch_size)

        # Train on this batch of images
        session.run(optimizer, feed_dict = {x: x_batch, y_loss: y_batch})

    # Compute the accuracy using test data
    predicted = tf.equal(tf.argmax(y, 1), tf.argmax(y_loss, 1))
    accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))
    print('\nAccuracy =', session.run(accuracy, feed_dict = {
            x: mnist.test.images, 
            y_loss: mnist.test.labels}))


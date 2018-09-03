import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
from Week23.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39
loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss
init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                               # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss


a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
sess = tf.Session()
print(sess.run(c))
x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()

# GRADED FUNCTION: linear_function

def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """
    np.random.seed(1)
    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name  = "b")
    Y = tf.add(tf.matmul(W, X), b)
    ### END CODE HERE ###
    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    ### START CODE HERE ###
    sess = tf.Session()
    result = sess.run(Y)
    ### END CODE HERE ###
    # close the session
    sess.close()
    return result

print( "result = " + str(linear_function()))

# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Computes the sigmoid of z
    Arguments:
    z -- input value, scalar or vector
    Returns:
    results -- the sigmoid of z
    """
    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name = "x")
    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)
    # Create a session, and run it. Please use the method 2 explained above.
    # You should use a feed_dict to pass z's value to x.
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict = {x:z})
    ### END CODE HERE ###
    return result
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))

# GRADED FUNCTION: cost

def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.

    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    ### START CODE HERE ###
    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
    z = tf.placeholder(tf.float32, name = "z")
    y = tf.placeholder(tf.float32, name = "y")
    # Use the loss function (approx. 1 line)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
    # Create a session (approx. 1 line). See method 1 above.
    sess = tf.Session()
    # Run the session (approx. 1 line).
    cost = sess.run(cost, feed_dict = {z:logits, y:labels})
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    ### END CODE HERE ###
    return cost

logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))

# GRADED FUNCTION: one_hot_matrix

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
    corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
    will be 1.
    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension
    Returns:
    one_hot -- one hot matrix
    """
    ### START CODE HERE ###
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(value = C, name = "C")
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    # Create the session (approx. 1 line)
    sess = tf.Session()
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    ### END CODE HERE ###
    return one_hot
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = " + str(one_hot))


# GRADED FUNCTION: ones
def ones(shape):
    """
    Creates an array of ones of dimension shape
    Arguments:
    shape -- shape of the array you want to create
    Returns:
    ones -- array containing only ones
    """
    ### START CODE HERE ###
    # Create "ones" tensor using tf.ones(...). (approx. 1 line)
    ones = tf.ones(shape)
    # Create the session (approx. 1 line)
    sess = tf.Session()
    # Run the session to compute 'ones' (approx. 1 line)
    ones = sess.run(ones)
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    ### END CODE HERE ###
    return ones

print ("ones = " + str(ones([3])))






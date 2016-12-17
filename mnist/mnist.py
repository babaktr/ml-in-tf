import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np 
from stats import Stats

# Get MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

flags = tf.app.flags

flags.DEFINE_integer('minibatches', 1000, 'Number of minibatches to run the training on.')
flags.DEFINE_float('learning_rate', 0.5, 'Learning rate of the optimizer.')
flags.DEFINE_integer('average_summary', 100, 'How often to print an average summary.')
flags.DEFINE_string('optimizer', 'gradent_descent', 'If another optimizer should be used [adam, rmsprop]. Defaults to gradient descent')
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested')

settings = flags.FLAGS

def test_model(y, y_):
    print ' --- TESTING MODEL ---'
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy on test set: {}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

print('Starting session with: Minibatches: {} -- Learning Rate: {} -- Optimizer: {}'.format(settings.minibatches,
                                                                                                            settings.learning_rate, 
                                                                                                            settings.optimizer)) 

sess = tf.InteractiveSession()

# Input of two values
x = tf.placeholder(tf.float32, shape=[None, 784])
# Desired output of one value
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Randomly initialize weights of layer 1
W = tf.Variable(tf.truncated_normal([784, 10]))
# Initialize theta/bias of layer 1
b = tf.Variable(tf.zeros([10]))

# Output
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Objective/Error function 
# E = - SUM(y_ * log(y))
obj_function = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

if settings.optimizer.lower() == 'adam':
    # Adam Optimizer
    train_step = tf.train.AdamOptimizer(settings.learning_rate).minimize(obj_function)
elif settings.optimizer.lower() == 'rmsprop':
    # RMSProp
    train_step = tf.train.RMSPropOptimizer(settings.learning_rate).minimize(obj_function)
else: 
    # Gradient Descent
    train_step = tf.train.GradientDescentOptimizer(settings.learning_rate).minimize(obj_function)

init = tf.global_variables_initializer()
sess.run(init)

# Statistics summary writer
summary_dir = '../log/mnist-hidden{}-lr{}-minibatches{}-{}/'.format(10, settings.learning_rate, settings.minibatches, settings.optimizer)
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
stats = Stats(sess, summary_writer, 1)

err_array = []
for i in range (settings.minibatches): 
    # Run training
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, err = sess.run([train_step, obj_function],
                        feed_dict={x: batch_xs,
                                    y_: batch_ys})

    stats.update(err, i)
    err_array.append(err)
    if i % settings.average_summary == 0:
        # Print average
        print "Minibatch: {}, Error: {}".format(i, np.average(err_array))
        err_array = []

if settings.run_test:
    test_model(y, y_)


                
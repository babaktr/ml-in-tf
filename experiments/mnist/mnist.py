import sys
sys.path.append('../..')

import tensorflow as tf
import numpy as np 
from stats import Stats

# Get MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

flags = tf.app.flags

flags.DEFINE_integer('minibatches', 1000, 'Number of minibatches to run the training on.')
flags.DEFINE_float('learning_rate', 0.5, 'Learning rate of the optimizer.')
flags.DEFINE_integer('status_update', 100, 'How often to print an status update.')
flags.DEFINE_string('optimizer', 'gradent_descent', 'If another optimizer should be used [adam, rmsprop]. Defaults to gradient descent')
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested')

settings = flags.FLAGS

def test_model():
    print ' --- TESTING MODEL ---'
    print('Accuracy on test set: {}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

print('Starting session with: Minibatches: {} -- Learning Rate: {} -- Optimizer: {}'.format(settings.minibatches,
                                                                                            settings.learning_rate, 
                                                                                            settings.optimizer)) 

sess = tf.InteractiveSession()

# Input of two values
x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
# Desired output of one value
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='desired-output')

# Hidden layer 1 weights and bias
W = tf.Variable(tf.truncated_normal([784, 10]), name='weights-1')
b = tf.Variable(tf.zeros([10]), name='bias-1')

# Output
with tf.name_scope('output') as scope:
    y = tf.nn.softmax(tf.matmul(x, W) + b)

# Objective function - Cross Entropy
# E = - SUM(y_ * log(y))
with tf.name_scope('loss') as scope:
    obj_function = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

with tf.name_scope('train') as scope:
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
summary_dir = '../../logs/mnist-hidden{}-lr{}-minibatches{}-{}/'.format(10, settings.learning_rate, settings.minibatches, settings.optimizer)
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
stats = Stats(sess, summary_writer, 2)

# Specify how accuracy is measured
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range (settings.minibatches): 
    # Get minibatch
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # Run training
    _, loss = sess.run([train_step, obj_function],
                        feed_dict={x: batch_xs,
                                    y_: batch_ys})

    # Calculate batch accuracy
    acc = sess.run(accuracy, feed_dict={x: batch_xs, 
                                        y_: batch_ys})

    stats.update({'loss': loss, 'accuracy': acc, 'step': i})

    if i % settings.status_update == 0:
        # Print update
        print 'Minibatch: {}, Loss: {}, Accuracy: {}'.format(i, loss, acc)

if settings.run_test:
    test_model()


                
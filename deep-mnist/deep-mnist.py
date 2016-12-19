import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np 
from stats import Stats

# Get MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

flags = tf.app.flags

flags.DEFINE_integer('minibatches', 20000, 'Number of minibatches to run the training on.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate of the optimizer.')
flags.DEFINE_integer('average_summary', 100, 'How often to print an average summary.')
flags.DEFINE_string('optimizer', 'gradent_descent', 'If another optimizer should be used [adam, rmsprop]. Defaults to gradient descent')
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested')
flags.DEFINE_boolean('use_gpu', False, 'If it should run on GPU rather than CPU.')


settings = flags.FLAGS

def test_model():
    print ' '
    print ' --- TESTING MODEL ---'
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy on test set: {}'.format(accuracy.eval(feed_dict={x: mnist.test.images, 
                                                        y_: mnist.test.labels, 
                                                        keep_prob: 1.0})))

    #print('Accuracy on test set: {}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], 
                            padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], 
                       	padding='SAME')


print('Starting session with: Minibatches: {} -- Learning Rate: {} -- Optimizer: {}'.format(settings.minibatches,
                                                                                                            settings.learning_rate, 
                                                                                                            settings.optimizer)) 

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

sess = tf.InteractiveSession()

with tf.device(device):
    # Input
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
    # reshape to 28x28
    x_img = tf.reshape(x, [-1,28,28,1])
    # Desired output of one value
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='desired-output')

    # Convolutional layer 1 weights and bias
    W_conv1 = weight_variable([5, 5, 1, 32], name='w-conv-1')
    b_conv1 = bias_variable([32], name='b-conv-1')

    # First conv layer output
    with tf.name_scope('conv-1') as scope:
        h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)

    # First layer pooling
    with tf.name_scope('max-pool-1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    # Second layer conv weights and biases
    W_conv2 = weight_variable([5, 5, 32, 64], name='w-conv-2')
    b_conv2 = bias_variable([64], name='b-conv-2')

    # Second layer conv output
    with tf.name_scope('conv-2') as scope:
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second layer pooling
    with tf.name_scope('max-pool-2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer with weights and biases
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name='w-fc')
    b_fc1 = bias_variable([1024], name='b-fc')

    # Fully connected layer output
    with tf.name_scope('fully-connected') as scope:
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Drop out to reduec overfitting
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer weights and biases
    W_fc2 = weight_variable([1024, 10], name='w-out')
    b_fc2 = bias_variable([10], name='b-out')

    # Output
    with tf.name_scope('output') as scope:
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Objective/Error function - Cross Entropy
    with tf.name_scope('error') as scope:
        obj_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

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
    summary_dir = '../log/mnist-hidden{}-lr{}-minibatches{}-{}/'.format(10, settings.learning_rate, settings.minibatches, settings.optimizer)
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    stats = Stats(sess, summary_writer, 1)

    err_array = []
    for i in range (settings.minibatches): 
        # Run training
        batch = mnist.train.next_batch(50)
        #batch_xs, batch_ys = mnist.train.next_batch(100)
        _, err = sess.run([train_step, obj_function],
                            feed_dict={x: batch[0],
                                        y_: batch[1],
                                        keep_prob: 0.5})

        stats.update(err, i)
        err_array.append(err)
        if i % settings.average_summary == 0:
            # Print average
            print "Minibatch: {}, Error: {}".format(i, np.average(err_array))
            err_array = []

    if settings.run_test:
        test_model()


                    
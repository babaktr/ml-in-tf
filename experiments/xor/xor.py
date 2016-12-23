import sys
sys.path.append('../..')

import tensorflow as tf
import numpy as np 
from stats import Stats

flags = tf.app.flags

flags.DEFINE_integer('batches', 20000, 'Number of batches (epochs) to run the training on.')
flags.DEFINE_integer('hidden_nodes', 2, 'Number of nodes to use in the two hidden layers.')
flags.DEFINE_float('learning_rate', 0.05, 'Learning rate of the optimizer.')
flags.DEFINE_integer('status_update', 1000, 'How often to print an status update.')
flags.DEFINE_string('optimizer', 'gradent_descent', 'Specifices optimizer to use [adam, rmsprop]. Defaults to gradient descent')
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested')

settings = flags.FLAGS

def test_model():
    print ' --- TESTING MODEL ---'
    print('[0.0, 0.0] -- Prediction: {}'.format(sess.run(y, feed_dict={x: np.array([[0.0, 0.0]])})))
    print('[0.0, 1.0] -- Prediction: {}'.format(sess.run(y, feed_dict={x: np.array([[0.0, 1.0]])})))
    print('[1.0, 0.0] -- Prediction: {}'.format(sess.run(y, feed_dict={x: np.array([[1.0, 0.0]])})))
    print('[1.0, 1.0] -- Prediction: {}'.format(sess.run(y, feed_dict={x: np.array([[1.0, 1.0]])})))

print('Starting session with: Batches: {} -- Hidden Nodes: {} -- Learning Rate: {} -- Optimizer: {}'.format(settings.batches, 
                                                                                                            settings.hidden_nodes, 
                                                                                                            settings.learning_rate, 
                                                                                                            settings.optimizer)) 

sess = tf.InteractiveSession()

# Input of two values
x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
# Desired output of one value
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='desired-output')

# Hidden layer 1 weights and bias
W_1 = tf.Variable(tf.truncated_normal([2, settings.hidden_nodes]), name='weights-1')
b_1 = tf.Variable(tf.zeros([settings.hidden_nodes]), name='bias-1')

# Hidden layer 1's output
with tf.name_scope('hidden-layer-1') as scope:
    out_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)

# Hidden layer 2 weights and bias 
W_2 = tf.Variable(tf.truncated_normal([settings.hidden_nodes, 2]), name='weights-2')
b_2 = tf.Variable(tf.zeros([2]), name='bias-2')

# Hidden layer 2's output 
with tf.name_scope('hidden-layer-2') as scope:
    out_2 = tf.nn.sigmoid(tf.matmul(out_1, W_2) + b_2)

# Output layer weights and bias 
W_3 = tf.Variable(tf.truncated_normal([2,1]), name='weights-3')
b_3 = tf.Variable(tf.zeros([1]), name='bias-3')

# Output layer's output
with tf.name_scope('output-layer') as scope:
    y = tf.nn.sigmoid(tf.matmul(out_2, W_3) + b_3)

# Objective function 
# E = - 1/2 (y - y_)^2
with tf.name_scope('loss') as scope:
    obj_function = 0.5 * tf.reduce_sum(tf.sub(y, y_) * tf.sub(y, y_))

# Set optimizer
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

# Prepare training data
training_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
training_outputs = [[0.0], [1.0], [1.0], [0.0]]

init = tf.global_variables_initializer()
sess.run(init)

# Statistics summary writer
summary_dir = '../logs/xor-hidden{}-lr{}-batches{}-{}/'.format(settings.hidden_nodes, settings.learning_rate, settings.batches, settings.optimizer)
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
stats = Stats(sess, summary_writer, 1)

# Set up how to measure accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range (settings.batches): 
    # Run training
    _, loss = sess.run([train_step, obj_function],
                        feed_dict={x: np.array(training_inputs),
                                    y_: np.array(training_outputs)})

    stats.update({'loss': loss, 'step': i})
    if i % settings.status_update == 0:
        # Print update
        print "Batch: {}, Loss: {}".format(i, loss)

if settings.run_test:
    test_model()
           
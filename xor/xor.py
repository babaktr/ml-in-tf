import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np 
from stats import Stats

flags = tf.app.flags

flags.DEFINE_integer('epochs', 20000, 'Number of epochs (batches) to run the training on.')
flags.DEFINE_integer('hidden_nodes', 2, 'Number of nodes to use in the two hidden layers.')
flags.DEFINE_float('learning_rate', 0.05, 'Learning rate of the optimizer.')
flags.DEFINE_integer('average_summary', 1000, 'How often to print an average summary.')
flags.DEFINE_string('optimizer', 'gradent_descent', 'If another optimizer should be used [adam, rmsprop]. Defaults to gradient descent')
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested')

settings = flags.FLAGS

def test_model():
    print ' --- TESTING MODEL ---'
    print('[0.0, 0.0] -- Prediction: {}'.format(sess.run(y_, feed_dict={x: np.array([[0.0, 0.0]])})))
    print('[0.0, 1.0] -- Prediction: {}'.format(sess.run(y_, feed_dict={x: np.array([[0.0, 1.0]])})))
    print('[1.0, 0.0] -- Prediction: {}'.format(sess.run(y_, feed_dict={x: np.array([[1.0, 0.0]])})))
    print('[1.0, 1.0] -- Prediction: {}'.format(sess.run(y_, feed_dict={x: np.array([[1.0, 1.0]])})))

print('Starting session with: Epochs: {} -- Hidden Nodes: {} -- Learning Rate: {} -- Optimizer: {}'.format(settings.epochs, 
                                                                                                            settings.hidden_nodes, 
                                                                                                            settings.learning_rate, 
                                                                                                            settings.optimizer)) 

sess = tf.InteractiveSession()

# Input of two values
x = tf.placeholder(tf.float32, shape=[None, 2])
# Desired output of one value
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Randomly initialize weights of layer 1
W_1 = tf.Variable(tf.truncated_normal([2, settings.hidden_nodes]))
# Initialize theta/bias of layer 1
b_1 = tf.Variable(tf.zeros([settings.hidden_nodes]))

# Layer 1 output
out_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)

# Same for layer 2 (hidden layer 1)
W_2 = tf.Variable(tf.truncated_normal([settings.hidden_nodes, 2]))
b_2 = tf.Variable(tf.zeros([2]))

# Layer 2 output
out_2 = tf.nn.sigmoid(tf.matmul(out_1, W_2) + b_2)

# And layer 3 (hidden layer 2)
W_3 = tf.Variable(tf.truncated_normal([2,1]))
b_3 = tf.Variable(tf.zeros([1]))

# Output of one value
y = tf.nn.sigmoid(tf.matmul(out_2, W_3) + b_3)

# Objective/Error function 
# E = - 1/2 (y - y_)^2
obj_function = 0.5 * tf.reduce_sum(tf.sub(y, y_) * tf.sub(y, y_))

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
summary_dir = '../log/xor-hidden{}-lr{}-epochs{}-{}/'.format(settings.hidden_nodes, settings.learning_rate, settings.epochs, settings.optimizer)
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
stats = Stats(sess, summary_writer, 1)

err_array = []
for i in range (settings.epochs): 
    # Run training
    _, err = sess.run([train_step, obj_function],
                        feed_dict={x: np.array(training_inputs),
                                    y_: np.array(training_outputs)})

    stats.update(err, i)
    err_array.append(err)
    if i % settings.average_summary == 0:
        # Print average
        print "Epoch: {}, Error: {}".format(i, np.average(err_array))
        err_array = []

if settings.run_test:
    test_model()


                
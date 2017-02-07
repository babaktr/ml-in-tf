import tensorflow as tf
import numpy as np

class NeuralNetwork(object):
    def __init__(self, device, random_seed, state_size, action_size, hidden_layers, hidden_nodes, learning_rate, optimizer):
        self.sess = tf.Session()
        self.device = device

        with tf.device(self.device):
            # Set random seed
            tf.set_random_seed(random_seed)
            with tf.name_scope('input') as scope:
                # Action input batch with shape [?, action_size]
                self.a = tf.placeholder(tf.float32, [None, action_size], name='action-input')

                # State input 
                self.s = tf.placeholder(tf.float32, shape=[None, state_size], name='s-input')

                # Target Q-value batch with shape [?, 1]
                self.y = tf.placeholder(tf.float32, shape=[None, 1], name='target-q_value')

            previous_layer = self.s
            previous_size = state_size

            # Hidden layer 1
            out_array = []
            for n in range(hidden_layers):
                layer_name = 'hidden-layer-' + str(n+1)
                with tf.name_scope(layer_name) as scope:
                    # Hidden layer 1 weights and bias
                    W_name = 'weight-' + str(n+1)
                    W = tf.Variable(tf.random_uniform([previous_size, hidden_nodes]), name=W_name)
                    b_name = 'bias-' + str(n+1)
                    b = tf.Variable(tf.zeros([hidden_nodes]), name=b_name)

                    # Hidden layer output
                    output_name = 'hidden-layer-' + str(n+1) + 'out'
                    with tf.name_scope(output_name) as scope:
                        out = tf.nn.relu(tf.matmul(previous_layer, W) + b)
                        out_array.append(out)

                previous_layer = out
                previous_size = hidden_nodes


            # Ouptut layer
            with tf.name_scope('output') as scope:
                # Ouptut layer weights and bias 
                W_out = tf.Variable(tf.random_uniform([hidden_nodes, action_size]), name='weight-out')
                b_out = tf.Variable(tf.zeros(action_size), name='bias-out')

                # Output output 
                with tf.name_scope('output_value') as scope:
                    self.q_values = tf.matmul(out, W_out) + b_out

            with tf.name_scope('optimizer') as scope:
                self.lr = tf.Variable(0, name='learn_rate-input')

                with tf.name_scope('loss'):
                    target_q_value = tf.reduce_sum(tf.multiply(self.q_values, self.a), reduction_indices=1)
                    self.loss_function = tf.reduce_mean(tf.square(tf.subtract(self.y, target_q_value)))

                if optimizer.lower() == 'adam':
                    # Adam Optimizer
                    self.optimizer_function = tf.train.AdamOptimizer(initial_learning_rate)
                elif optimizer.lower() == 'gradientdecent':
                    # Gradient Descent
                    self.optimizer_function = tf.train.GradientDescentOptimizer(initial_learning_rate)
                else: 
                    # RMSProp
                    self.optimizer_function = tf.train.RMSPropOptimizer(self.lr)

                    
                with tf.name_scope('training_op') as scope:
                    self.train_op = self.optimizer_function.minimize(self.loss_function)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    '''
    Utilizes the optimizer and objectie function to train the network based on the input and target output.
    '''


    def train(self, s_input, a_input, y_input, learn_rate):
        with tf.device(self.device):
            _, loss = self.sess.run([self.train_op, self.loss_function], feed_dict={self.s: np.vstack(s_input), self.a: np.vstack(a_input), self.y: np.vstack(y_input), self.lr: learn_rate})
            return loss

    '''
    Feeds a value through the network and produces an output.
    '''
    def predict(self, s_input):
        with tf.device(self.device):
            predicted_output = self.sess.run(self.q_values, feed_dict={self.s: np.vstack(s_input)})
            return predicted_output
import tensorflow as tf
import numpy as np

class NeuralNetworks(object):
    def __init__(self, device, random_seed, input_size, output_size, hidden_size, p_learning_rate, v_learning_rate, optimizer):
        self.sess = tf.Session()

        with tf.device(device):
            # Set random seed
            tf.set_random_seed(random_seed)

            # State input 
            self.state = tf.placeholder(tf.float32, shape=[None, input_size], name='state-input') 

            self.p_network = self.policy_network(input_size, output_size, optimizer, p_learning_rate)
            self.v_network = self.value_network(input_size, hidden_size, optimizer, v_learning_rate)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def get_train_op(self, optimizer, learning_rate, loss):
        print optimizer
        if optimizer.lower() == 'adam':
            # Adam Optimizer
            return  tf.train.AdamOptimizer(learning_rate).minimize(loss)
        elif optimizer.lower() == 'rmsprop':
            # RMSProp
            return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        else:                 
            # Gradient Descent
            return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    def policy_network(self, input_size, output_size, optimizer, learning_rate):
        with tf.name_scope('policy_network') as scope:
            parameters = tf.get_variable('policy_parameters', [input_size, output_size])
            self.action = tf.placeholder(tf.float32, [None, output_size])
            self.target_advantage = tf.placeholder(tf.float32, [None, 1])
            linear_output = tf.matmul(self.state, parameters)

            #self.policy = tf.nn.softmax(linear_output)
            self.policy = tf.nn.softmax(linear_output)

            good_probabilities = tf.reduce_sum(tf.multiply(self.policy, self.action),reduction_indices=[1])
            log_probabilities = tf.log(good_probabilities)

            eligibility = log_probabilities * self.target_advantage
            
            #self.policy_loss = -tf.reduce_sum(log_probabilities)

            with tf.name_scope('loss') as scope:
                #self.policy_loss = tf.negative(tf.reduce_sum(tf.multiply(tf.log(self.policy), self.action)))
                self.policy_loss = -tf.reduce_sum(eligibility)
            with tf.name_scope('optimizer') as scope:
                self.policy_train_op = self.get_train_op(optimizer, learning_rate, self.policy_loss)

    def value_network(self, input_size, hidden_size, optimizer, learning_rate):
        with tf.name_scope('value_network') as scope:
            self.target_value = tf.placeholder(tf.float32, [None, 1])

            # Hidden layer
            with tf.name_scope('hidden-layer') as scope:
                # Hidden layer 1 weights and bias
                W = tf.get_variable('weights', [input_size, hidden_size])
                b = tf.get_variable('bias', [hidden_size])

                # Hidden layer output
                with tf.name_scope('hidden-layer-out') as scope:
                    h1 = tf.nn.relu(tf.matmul(self.state, W) + b)

            with tf.name_scope('output') as scope:
                # Ouptut layer weights and bias 
                W_out = tf.get_variable('weights-out', [hidden_size, 1])
                b_out = tf.get_variable('bias-out', [1])

                # Output 
                with tf.name_scope('output_value') as scope:
                    self.value = tf.matmul(h1, W_out) + b_out

            with tf.name_scope('loss') as scope:
                difference = tf.subtract(self.value, self.target_value)
                self.value_loss = tf.nn.l2_loss(difference)

            with tf.name_scope('optimizer') as scope:
                self.value_train_op = self.get_train_op(optimizer, learning_rate, self.value_loss)


    def predict_policy(self, state_input):
        return self.sess.run(self.policy, feed_dict={self.state: [state_input]})

    def train_policy(self, state_input, action, target_a):
        _, loss = self.sess.run([self.policy_train_op, self.policy_loss],
                    feed_dict={self.state: np.vstack(state_input),
                            self.action: np.vstack(action),
                            self.target_advantage: np.vstack(target_a)})
        return loss

    def predict_value(self, state_input):
        return self.sess.run(self.value, feed_dict={self.state: [state_input]})

    def train_value(self, state_input, target_v):
        _, loss = self.sess.run([self.value_train_op, self.value_loss],
                    feed_dict={self.state: np.vstack(state_input),
                            self.target_value: np.vstack(target_v)})
        return loss

import tensorflow as tf

class RecurrentNeuralNetwork(object):
    def __init__(self, device, random_seed, input_size, hidden, sequence_length, rnn_layers, learning_rate, optimizer):
        self.sess = tf.InteractiveSession()

        with tf.device(device):
            # Set random seed
            tf.set_random_seed(random_seed)

            # Input with shape 
            self.x = tf.placeholder(tf.float32, shape=[sequence_length, None, input_size], name='x-input')
            # Desired output
            self.y_ = tf.placeholder(tf.float32, shape=[None, 4], name='desired-output')

            # Weights and bias
            W = tf.Variable(tf.random_uniform([hidden, 4]), name='weights')
            b = tf.Variable(tf.random_uniform([4]), name='bias')

            with tf.name_scope('pre-process'):
                # Reshape input from [3, ?, input_size]
                x_1 = tf.reshape(self.x, [-1, input_size])
                # Split input to shape [3, [?, input_size]]
                x_2 = tf.split(0, sequence_length, x_1)

            # Set up basic LSTM cell with forget_bias = 1
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * rnn_layers, state_is_tuple=True)

            # Get outputs of RNN layer
            outputs, state = tf.nn.rnn(lstm_cell, x_2, dtype=tf.float32)

            with tf.name_scope('output') as scope:
                self.y = tf.matmul(outputs[-1], W) + b

            # Objective function 
            with tf.name_scope('loss') as scope:
                self.obj_function = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.y_, self.y))))

            with tf.name_scope('train') as scope:
                if optimizer.lower() == 'adam':
                    # Adam Optimizer
                    self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.obj_function)
                elif optimizer.lower() == 'rmsprop':
                    # RMSProp
                    self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(self.obj_function)
                else: 
                    # Gradient Descent
                    self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.obj_function)

            init = tf.global_variables_initializer()
            self.sess.run(init)

            # Specify how accuracy is measured
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    '''
    Utilizes the optimizer and objectie function to train the network based on the input and desired output.
    '''
    def train(self, x_input, desired_output):
        _, loss = self.sess.run([self.train_step, self.obj_function],
                    feed_dict={self.x: x_input,
                            self.y_: desired_output})
        return loss

    '''
    Feeds a value through the network and produces an output.
    '''
    def predict(self, x_input):
        predicted_output = self.sess.run(self.y, feed_dict={self.x: x_input})
        return predicted_output

    '''
    Measures the accuracy of the network based on the specified accuracy measure, the input and the desired output.
    '''
    def get_accuracy(self, x_input, desired_output):
        acc = self.sess.run(self.accuracy, feed_dict={self.x: x_input, 
                                            self.y_: desired_output})
        return acc

import tensorflow as tf

class NeuralNetwork(object):
    def __init__(self, random_seed, hidden_n, learning_rate, optimizer):
        self.sess = tf.InteractiveSession()

        # Set random seed
        tf.set_random_seed(random_seed)

        # Input of two values
        with tf.name_scope('input') as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')

        # Target output of one value
        with tf.name_scope('target_output') as scope: 
            self.y_ = tf.placeholder(tf.float32, shape=[None, 1], name='target-output')

        # Hidden layer 1
        with tf.name_scope('hidden-layer-1') as scope: 
            # Hidden layer 1 weights and bias
            W_1 = tf.Variable(tf.truncated_normal([2, hidden_n]), name='weights-1')
            b_1 = tf.Variable(tf.zeros([hidden_n]), name='bias-1')

            # Hidden layer 1's output
            with tf.name_scope('hidden-layer-1-out') as scope:
                out_1 = tf.nn.sigmoid(tf.matmul(self.x, W_1) + b_1)

        # Hidden layer 2
        with tf.name_scope('hidden-layer-2') as scope: 
            # Hidden layer 2 weights and bias 
            W_2 = tf.Variable(tf.truncated_normal([hidden_n, 2]), name='weights-2')
            b_2 = tf.Variable(tf.zeros([2]), name='bias-2')

            # Hidden layer 2's output 
            with tf.name_scope('hidden-layer-2-out') as scope:
                out_2 = tf.nn.sigmoid(tf.matmul(out_1, W_2) + b_2)

        # Output layer
        with tf.name_scope('output') as scope:
            # Output layer weights and bias  
            W_3 = tf.Variable(tf.truncated_normal([2,1]), name='weights-3')
            b_3 = tf.Variable(tf.zeros([1]), name='bias-3')

            # Output layer's output
            with tf.name_scope('output_value') as scope:
                self.y = tf.nn.sigmoid(tf.matmul(out_2, W_3) + b_3)

        # Objective function 
        # E = - 1/2 (y - y_)^2
        with tf.name_scope('loss') as scope:
            self.obj_function = tf.multiply(0.5, tf.reduce_sum(tf.subtract(self.y, self.y_) * tf.subtract(self.y, self.y_)))

        # Set optimizer
        with tf.name_scope('optimizer') as scope:
            if optimizer.lower() == 'adam':
                # Adam Optimizer
                self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.obj_function)
            elif optimizer.lower() == 'rmsprop':
                # RMSProp
                self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(self.obj_function)
            else: 
                # Gradient Descent
                self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.obj_function)

        with tf.name_scope('accuracy'):
            # Specify how accuracy is measured
            correct_prediction = tf.subtract(1.0, tf.abs(tf.subtract(self.y_, self.y)))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    '''
    Utilizes the optimizer and objectie function to train the network based on the input and target output.
    '''
    def train(self, x_input, target_output):
        _, loss = self.sess.run([self.train_step, self.obj_function],
                    feed_dict={self.x: x_input,
                            self.y_: target_output})
        return loss

    '''
    Feeds a value through the network and produces an output.
    '''
    def predict(self, x_input):
        predicted_output = self.sess.run(self.y, feed_dict={self.x: x_input})
        return predicted_output

    '''
    Measures the accuracy of the network based on the specified accuracy measure, the input and the target output.
    '''
    def get_accuracy(self, x_input, target_output):
        acc = self.sess.run(self.accuracy, feed_dict={self.x: x_input, 
                                                    self.y_: target_output})
        return acc

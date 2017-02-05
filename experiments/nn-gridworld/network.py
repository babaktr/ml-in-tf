import tensorflow as tf

class NeuralNetwork(object):
    def __init__(self, device, random_seed, input_size, hidden_l1, hidden_l2, learning_rate, optimizer):
        self.sess = tf.Session()

        with tf.device(device):
            # Set random seed
            tf.set_random_seed(random_seed)

            # Input with shape [?, input_size]
            with tf.name_scope('input') as scope:
                self.x = tf.placeholder(tf.float32, shape=[None, input_size], name='x-input')

            # Target output with shape [?, 4]
            with tf.name_scope('target_output') as scope:
                self.y_ = tf.placeholder(tf.float32, shape=[None, 4], name='target-output')

            # Hidden layer 1
            with tf.name_scope('hidden-layer-1') as scope:
                # Hidden layer 1 weights and bias
                W_1 = tf.Variable(tf.random_uniform([input_size, hidden_l1]), name='weights-1')
                b_1 = tf.Variable(tf.zeros([hidden_l1]), name='bias-1')

                # Hidden layer 1's output
                with tf.name_scope('hidden-layer-1-out') as scope:
                    out_1 = tf.nn.relu(tf.matmul(self.x, W_1) + b_1)

            # Hidden layer 2
            with tf.name_scope('hidden-layer-2') as scope:
                # Hidden layer 2 weights and bias 
                W_2 = tf.Variable(tf.random_uniform([hidden_l1, hidden_l2]), name='weights-2')
                b_2 = tf.Variable(tf.zeros([hidden_l2]), name='bias-2')

                # Hidden layer 2's output 
                with tf.name_scope('hidden-layer-2-out') as scope:
                    out_2 = tf.nn.relu(tf.matmul(out_1, W_2) + b_2)

            # Ouptut layer
            with tf.name_scope('output') as scope:
                # Ouptut layer weights and bias 
                W_3 = tf.Variable(tf.random_uniform([hidden_l2, 4]), name='weights-3')
                b_3 = tf.Variable(tf.zeros([4]), name='bias-3')

                # Hidden layer 3's output 
                with tf.name_scope('output_value') as scope:
                    self.y = tf.matmul(out_2, W_3) + b_3

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

            # Specify how accuracy is measured
            with tf.name_scope('accuracy') as scope:
                correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
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
    Measures the accuracy of the network based on the specified accuracy measure, the input and the desired output.
    '''
    def get_accuracy(self, x_input, target_output):
        acc = self.sess.run(self.accuracy, feed_dict={self.x: x_input, 
                                                    self.y_: target_output})
        return acc
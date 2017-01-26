import tensorflow as tf

class ConvolutionalNeuralNetwork(object):
    '''
    Set up weight variable.
    '''
    def weight_variable(self, shape, name):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name=name)

    '''
    Set up bias variable.
    '''
    def bias_variable(self, shape, name):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name=name)

    '''
    Set up 2D convolution.
    '''
    def conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], 
                                padding='SAME')

    '''
    Set up 2x2 max pooling.
    '''
    def max_pool_2x2(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], 
                           	padding='SAME')

    def __init__(self, device, random_seed, learning_rate, optimizer):
        self.sess = tf.InteractiveSession()

        with tf.device(device):
            # Set random seed
            tf.set_random_seed(random_seed)

            # Input
            self.x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
            # reshape to 28x28
            x_img = tf.reshape(self.x, [-1,28,28,1])
            # Desired output of one value
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='desired-output')

            # Convolutional layer 1 weights and bias
            with tf.name_scope('conv1') as scope
                W_conv1 = self.weight_variable([5, 5, 1, 32], name='w-conv1')
                b_conv1 = self.bias_variable([32], name='b-conv1')

                # First conv layer output
                with tf.name_scope('conv1-out') as scope:
                    h_conv1 = tf.nn.relu(self.conv2d(x_img, W_conv1) + b_conv1)

            # First layer pooling
            with tf.name_scope('max-pool1') as scope:
                h_pool1 = self.max_pool_2x2(h_conv1)

            # Second layer conv weights and biases
            with tf.name_scope('conv2') as scope
                W_conv2 = self.weight_variable([5, 5, 32, 64], name='w-conv2')
                b_conv2 = self.bias_variable([64], name='b-conv2')

                # Second layer conv output
                with tf.name_scope('conv2-out') as scope:
                    h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

            # Second layer pooling
            with tf.name_scope('max-pool2') as scope:
                h_pool2 = self.max_pool_2x2(h_conv2)

            # Fully connected layer with weights and biases
            with tf.name_scope('fully-connected') as scope:
                W_fc1 = self.weight_variable([7 * 7 * 64, 1024], name='w-fc')
                b_fc1 = self.bias_variable([1024], name='b-fc')

                # Fully connected layer output
                with tf.name_scope('fully-connected-out') as scope:
                    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
                    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # Drop out to reduce overfitting
            self.keep_prob = tf.placeholder(tf.float32, name='keep-prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            # Output layer weights and biases
            with tf.name_scope('output') as scope:
                W_fc2 = self.weight_variable([1024, 10], name='w-out')
                b_fc2 = self.bias_variable([10], name='b-out')

                # Output
                with tf.name_scope('classification') as scope:
                    self.y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            # Objective/Error function - Cross Entropy
            with tf.name_scope('loss') as scope:
                self.obj_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))

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
            with tf.name_scope('accuracy') as scope:
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    '''
    Utilizes the optimizer and objectie function to train the network based on the input and desired output.
    '''
    def train(self, x_input, desired_output):
        _, loss = self.sess.run([self.train_step, self.obj_function],
                                feed_dict={self.x: x_input,
                                        self.y_: desired_output,
                                        self.keep_prob: 0.5})
        return loss

    '''
    Feeds a value through the network and produces an output.
    '''
    def predict(self, x_input):
        predicted_output = self.sess.run(self.y, 
                                        feed_dict={self.x: x_input,
                                                self.keep_prob: 1.0})
        return predicted_output

    '''
    Measures the accuracy of the network based on the specified accuracy measure, the input and the desired output.
    '''
    def get_accuracy(self, x_input, desired_output):
        acc = self.sess.run(self.accuracy, 
                            feed_dict={self.x: x_input, 
                                    self.y_: desired_output,
                                    self.keep_prob: 1.0})
        return acc

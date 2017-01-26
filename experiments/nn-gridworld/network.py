import tensorflow as tf

class ConvolutonalNeuralNetwork(object):
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
    def conv2d(self, x, W, stride):
      return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], 
                                padding='VALID')

    def __init__(self, index, device, random_seed, input_size, action_size, hidden_l1, hidden_l2, learning_rate, optimizer):
        self.sess = tf.InteractiveSession()

        with tf.device(device):
            # Set random seed
            tf.set_random_seed(random_seed)

            # Input with shape [?, input_size]
            self.x = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name='x_t-input')
            # Desired output with shape [?, action_size]
            self.y_ = tf.placeholder(tf.float32, shape=[None, action_size], name='desired-output')

            # Convolutional layer 1 weights and bias with stride=4, produces 16 19x19 outputs
            W_conv1 = self.weight_variable([8, 8, 4, 16], 'w_conv1')
            b_conv1 = self.bias_variable([16]'bias-1')
            stride_1 = tf.Variable(4, name='stride_1')

            # First conv layer output
             with tf.name_scope('conv-1') as scope:
                h_1 = tf.nn.relu(tf.conv2d(self.x, W_conv, stride_1) + b_conv1)

            # Second layer conv weights and biases with stride=2, produces 32 9x9 outputs
            W_conv2 = self.weight_variable([4, 4, 16, 32], name='w-conv2')
            b_conv2 = self.bias_variable([32], name='b-conv2')
            stride_2 = tf.Variable(2, name='stride_2')


            # Convolutional layer 2's output 
            with tf.name_scope('conv-2') as scope:
                out_2 = tf.nn.relu(self.conv2d(out_1, W_conv2, stride_2) + b_conv2)

            # 256 Fully connected units with weights and biases
            W_fc1 = self.weight_variable([9*9*32, 256], name='w-fc')
            b_fc1 = self.bias_variable([256], name='b-fc')

            # Fully connected layer output
            with tf.name_scope('fully-connected') as scope:
                h_conv2_flat = tf.reshape(h_pool2, [-1, 9*9*32])
                h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

            # Output layer weights and biases
            W_fc2 = self.weight_variable([256, action_size], name='w-out')
            b_fc2 = self.bias_variable([action_size], name='b-out')

            # Output
            with tf.name_scope('output') as scope:
                self.y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            # Objective function 
            with tf.name_scope('loss') as scope:
                self.obj_function = tf.reduce_mean(tf.square(self.y_ - self.y))

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

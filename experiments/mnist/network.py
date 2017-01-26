import tensorflow as tf

class NeuralNetwork(object):
    def __init__(self, device, random_seed, learning_rate, optimizer):
        self.sess = tf.InteractiveSession()

        with tf.device(device):
            # Set random seed
            tf.set_random_seed(random_seed)

            # Input with shape [?, 784]
            self.x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
            # Target output with shape [?, 10]
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='target-output')

            # Hidden layer 1 weights and bias
            W = tf.Variable(tf.truncated_normal([784, 10]), name='weights-1')
            b = tf.Variable(tf.zeros([10]), name='bias-1')

            # Output with shape [?, 10]
            with tf.name_scope('output') as scope:
                self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)

            # Objective function - Cross Entropy
            # E = - SUM(y_ * log(y))
            with tf.name_scope('loss') as scope:
                self.obj_function = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

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
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
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

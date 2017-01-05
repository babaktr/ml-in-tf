import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('batches', 10000, 'Number of batches (epochs) to run the training on.')
flags.DEFINE_integer('hidden_n', 2, 'Number of nodes to use in the two hidden layers.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate of the optimizer.')
flags.DEFINE_integer('status_update', 1000, 'How often to print an status update.')
flags.DEFINE_string('optimizer', 'gradent_descent', 'Specifices optimizer to use [adam, rmsprop]. Defaults to gradient descent')
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested.')

settings = flags.FLAGS

def test_model():
    print ' --- TESTING MODEL ---'
    print('[0.0, 0.0] -- Prediction: {}'.format(sess.run(y, feed_dict={x: np.array([[0.0, 0.0]])})))
    print('[0.0, 1.0] -- Prediction: {}'.format(sess.run(y, feed_dict={x: np.array([[0.0, 1.0]])})))
    print('[1.0, 0.0] -- Prediction: {}'.format(sess.run(y, feed_dict={x: np.array([[1.0, 0.0]])})))
    print('[1.0, 1.0] -- Prediction: {}'.format(sess.run(y, feed_dict={x: np.array([[1.0, 1.0]])})))

print('Starting session with: Batches: {} -- Hidden Neurons: {} -- Learning Rate: {} -- Optimizer: {}'.format(settings.batches, 
                                                                                                            settings.hidden_n, 
                                                                                                            settings.learning_rate, 
                                                                                                            settings.optimizer)) 

class NeuralNetwork(object):
    def __init__(self, random_seed, hidden_n, learning_rate, optimizer)
        self.sess = tf.InteractiveSession()

        # Input of two values
        self.x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
        # Desired output of one value
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1], name='desired-output')

        # Hidden layer 1 weights and bias
        W_1 = tf.Variable(tf.truncated_normal([2, settings.hidden_n]), name='weights-1')
        b_1 = tf.Variable(tf.zeros([settings.hidden_n]), name='bias-1')

        # Hidden layer 1's output
        with tf.name_scope('hidden-layer-1') as scope:
            out_1 = tf.nn.sigmoid(tf.matmul(self.x, W_1) + b_1)

        # Hidden layer 2 weights and bias 
        W_2 = tf.Variable(tf.truncated_normal([settings.hidden_n, 2]), name='weights-2')
        b_2 = tf.Variable(tf.zeros([2]), name='bias-2')

        # Hidden layer 2's output 
        with tf.name_scope('hidden-layer-2') as scope:
            out_2 = tf.nn.sigmoid(tf.matmul(out_1, W_2) + b_2)

        # Output layer weights and bias 
        W_3 = tf.Variable(tf.truncated_normal([2,1]), name='weights-3')
        b_3 = tf.Variable(tf.zeros([1]), name='bias-3')

        # Output layer's output
        with tf.name_scope('output-layer') as scope:
            self.y = tf.nn.sigmoid(tf.matmul(out_2, W_3) + b_3)

        # Objective function 
        # E = - 1/2 (y - y_)^2
        with tf.name_scope('loss') as scope:
            self.obj_function = tf.multiply(0.5, tf.reduce_sum(tf.sub(self.y, self.y_) * tf.sub(self.y, self.y_)))

        # Set optimizer
        with tf.name_scope('train') as scope:
            if optimizer.lower() == 'adam':
                # Adam Optimizer
                self.train_step = tf.train.AdamOptimizer(settings.learning_rate).minimize(obj_function)
            elif optimizer.lower() == 'rmsprop':
                # RMSProp
                self.train_step = tf.train.RMSPropOptimizer(settings.learning_rate).minimize(obj_function)
            else: 
                # Gradient Descent
                self.train_step = tf.train.GradientDescentOptimizer(settings.learning_rate).minimize(obj_function)

        with tf.name_scope('accuracy'):
            # Specify how accuracy is measured
            correct_prediction = tf.subtract(1.0, tf.abs(tf.subtract(self.y_, self.y)))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
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

import tensorflow as tf

class RecurrentNeuralNetwork(object):
    def __init__(self, device, random_seed, hidden_size, input_size, sequence_length, batch_size, learning_rate, optimizer):
        self.sess = tf.Session()

        with tf.device(device):
            # Set random seed
            tf.set_random_seed(random_seed)

            # Input with shape [?, 28, 28]
            with tf.name_scope('input') as scope:
                self.x = tf.placeholder(tf.float32, shape=[None, sequence_length*input_size], name='x-input')
                with tf.name_scope('input-preprocess'):
                    # Get x input in correct format [batch_size, ]
                    x_squared = tf.reshape(self.x, [-1, settings.sequence_length, settings.input_size])
                    x_transposed = tf.transpose(x_squared, [1,0,2])
                    x_reshaped = tf.reshape(x_transposed, [-1, input_size])
                    x_split = tf.split(x_reshaped, num_or_size_splits=sequence_length, axis=0)
                
                # Target input with shape [?, 10]
                with tf.name_scope('target_input') as scope:
                    self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='target-output')

            # Hidden layer 1
           
                # Hidden layer 1 weights and bias
                
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            #state = tf.zeros([batch_size, lstm_cell.state_size])
            with tf.name_scope('initial_state') as scope:
                state = lstm_cell.zero_state(batch_size, dtype=tf.float32)


            #state = lstm_cell.zero_state(batch_size, tf.float32)
            outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, x_split, state)
            #print 'outputs.shape: {}'.format(outputs.get_shape())
            #print 'states.shape: {}'.format(states.get_shape())
            # Output with shape [?, 10]
            with tf.name_scope('output') as scope:
                W = tf.Variable(tf.truncated_normal([hidden_size, 10]), name='weights-1')
                b = tf.Variable(tf.zeros([10]), name='bias-1')
                self.y = tf.matmul(outputs[-1], W) + b

            # Objective function - Cross Entropy
            # E = - SUM(y_ * log(y))
            
            with tf.name_scope('optimizer') as scope:

                with tf.name_scope('loss') as scope:
                    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
                    #self.obj_function = tf.reduce_mean(tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
                    #
                
                if optimizer.lower() == 'adam':
                    # Adam Optimizer
                    self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                elif optimizer.lower() == 'rmsprop':
                    # RMSProp
                    self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
                else: 
                    # Gradient Descent
                    self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)


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
        _, loss = self.sess.run([self.train_step, self.loss],
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

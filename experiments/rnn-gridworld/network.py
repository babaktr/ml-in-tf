import tensorflow as tf

class RecurrentNeuralNetwork(object):
    def __init__(self, device, random_seed, input_size, hidden, sequence_length, rnn_layers, learning_rate, optimizer):
        self.sess = tf.InteractiveSession()

        with tf.device(device):
            # Set random seed
            tf.set_random_seed(random_seed)

            # Input with shape 
            with tf.name_scope('input') as scope:
                self.x = tf.placeholder(tf.float32, shape=[sequence_length, input_size], name='x-input')
                print 'x initial shape: {}'.format(self.x.get_shape())
            # Target output
            with tf.name_scope('target_output') as scope:
                self.y_ = tf.placeholder(tf.float32, shape=[sequence_length, 4], name='target-output')

            with tf.name_scope('pre-process'):
                # Permuting batch_size and n_steps
                #self.x = tf.transpose(self.x, [1, 0, 2])
                # Reshaping to (n_steps*batch_size, n_input)
                #self.x = tf.reshape(self.x, [-1, input_size])
                # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
                

                #self.x_split = tf.split(self.x, num_or_size_splits=sequence_length, axis=0)
                
                self.x_split = tf.unstack(self.x, axis=0)
                print 'split x: {}'.format(tf.shape(self.x_split))
                

                #self.x_split = self.x_split
                

            # RNN
            with tf.name_scope('RNN') as scope:

                init_state = tf.placeholder(tf.float32, [rnn_layers, 2, hidden])

                state_per_layer_list = tf.unpack(init_state, axis=0)
                rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(rnn_layers)])


                # LSTM Cell
                with tf.name_scope('LSTM_cell') as scope:

                    self.state = np.zeros((rnn_layers, 2, state_size))
                    # Set up basic LSTM cell with forget_bias = 1
                    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden, forget_bias=1.0, state_is_tuple=True)
                    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * rnn_layers, state_is_tuple=True)
                    #lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * rnn_layers, state_is_tuple=True)

                    # Get outputs of LSTM layer
                    #initial_state = state = stacked_lstm.zero_state([rnn_layers, 2, input_size, tf.float32]) 
                    #print 'initial_state shape: {}'.format(initial_s)                 
                    with tf.name_scope('LSTM-out'):
                        #outputs, state = tf.nn.rnn(lstm_cell, x_2, dtype=tf.float32)
                        
                        outputs, self.state = stacked_lstm(self.x, initial_state=rnn_tuple_state)
                        print 'state shape: {}'.format(state)
                        print 'outputs shape: {}'.format(outputs)                        
                        print '--'

                        #final_state = state
                        print tf.shape(final_state)
            #self.y = outputs
            #print 'output shape: {}'.format(outputs.shape)
            #print 'output[-1] shape: {}'.format(outputs[-1].shape)
            with tf.name_scope('output') as scope:
                W = tf.Variable(tf.random_uniform([hidden, 4]), name='weights')
                b = tf.Variable(tf.random_uniform([4]), name='bias')
                self.y = [tf.matmul(outputs, W) + b for output in outputs]
                print self.y
                print tf.shape(self.y)

            targets = tf.unstack(self.y_, axis=0)
            # Objective function 
            with tf.name_scope('loss') as scope:
                losses = [tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value)))) for target, value in zip(targets, self.y)]
                self.total_loss = tf.reduce_mean(losses)

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
    Measures the accuracy of the network based on the specified accuracy measure, the input and the target output.
    '''
    def get_accuracy(self, x_input, target_output):
        acc = self.sess.run(self.accuracy, feed_dict={self.x: x_input, 
                                                    self.y_: target_output})
        return acc

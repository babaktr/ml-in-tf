import tensorflow as tf
import numpy as np

class DeepQNetwork(object):
    def __init__(self, sess, device, name, random_seed, action_size, trainable=False, gradient_clip_norm=40.):
        self.sess = sess
        self.device = device
        self.action_size = action_size
        tf.set_random_seed(random_seed)
        with tf.name_scope(name) as scope:
            self.build_network(trainable, gradient_clip_norm)

    '''
    Set up convolutional weight variable.
    '''
    def conv_weight_variable(self, shape, name):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        return tf.Variable(initializer(shape=shape), name=name)

    '''
    Set up weight variable.
    '''
    def weight_variable(self, shape, name):
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape=shape), name=name)
    
    '''
    Set up bias variable.
    '''
    def bias_variable(self, shape, name):
        init_value = tf.constant(0.0, shape=shape)
        return tf.Variable(init_value, name=name)

    '''
    Set up 2D convolution.
    '''
    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


    def build_network(self, trainable, gradient_clip_norm):

        with tf.name_scope('input') as scope:
            # State input batch with shape [?, 84, 84, 4]
            self.s = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name='s-input')

            if trainable:
                # Action input batch with shape [?, action_size]
                self.a = tf.placeholder(tf.float32, [None, self.action_size], name='action-input')

                # Target Q-value batch with shape [?, 1]
                self.y = tf.placeholder(tf.float32, shape=[None, 1], name='target-q_value')

        if True: #NIPS
            with tf.name_scope('conv-1') as scope:
                self.W_conv1 = self.conv_weight_variable([8, 8, 4, 16], name='w_conv1')
                self.b_conv1 = self.bias_variable([16], name='b_conv1')
                stride_1 = 4

                with tf.name_scope('conv1-out') as scope:
                    self.h_conv1 = tf.nn.relu(tf.add(self.conv2d(self.s, self.W_conv1, stride_1), self.b_conv1))

            with tf.name_scope('conv-2') as scope:
                self.W_conv2 = self.conv_weight_variable([4, 4, 16, 32], name='w_conv2')
                self.b_conv2 = self.bias_variable([32], name='b_conv2')
                stride_2 = 2

                with tf.name_scope('conv2-out') as scope:
                    self.h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_conv1, self.W_conv2, stride_2), self.b_conv2))

            with tf.name_scope('fully_connected') as scope:
                
                self.W_fc1 = self.weight_variable([9*9*32, 256], name='w_fc')
                self.b_fc1 = self.bias_variable([256], name='b_fc')

                # Fully connected layer output
                with tf.name_scope('fully-connected-out') as scope:
                    h_conv2_flat = tf.reshape(self.h_conv2, [tf.negative(1), 9*9*32])
                    self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_conv2_flat, self.W_fc1), self.b_fc1))

            # Output layer weights and biases
            with tf.name_scope('output') as scope:
                self.W_fc2 = self.weight_variable([256, self.action_size], name='w_out')
                self.b_fc2 = self.bias_variable([self.action_size], name='b_out')

                # Output
                with tf.name_scope('q_values') as scope:
                    self.q_values = tf.add(tf.matmul(self.h_fc1, self.W_fc2), self.b_fc2)

        else:
            with tf.name_scope('conv-1') as scope:
                self.W_conv1 = self.conv_weight_variable([8, 8, 4, 32], name='w_conv1')
                self.b_conv1 = self.bias_variable([32], name='b_conv1')
                stride_1 = 4

                with tf.name_scope('conv1-out') as scope:
                    self.h_conv1 = tf.nn.relu(tf.add(self.conv2d(self.s, self.W_conv1, stride_1), self.b_conv1))

            with tf.name_scope('conv-2') as scope:
                self.W_conv2 = self.conv_weight_variable([4, 4, 32, 64], name='w_conv2')
                self.b_conv2 = self.bias_variable([64], name='b_conv2')
                stride_2 = 2

                with tf.name_scope('conv2-out') as scope:
                    self.h_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_conv1, self.W_conv2, stride_2), self.b_conv2))

            with tf.name_scope('conv-3') as scope:
                self.W_conv3 = self.conv_weight_variable([3, 3, 64, 64], name='w_conv3')
                self.b_conv3 = self.bias_variable([64], name='b_conv3')
                stride_3 = 1

                with tf.name_scope('conv3-out') as scope:
                    self.h_conv3 = tf.nn.relu(tf.add(self.conv2d(self.h_conv2, self.W_conv3, stride_3), self.b_conv3))

            with tf.name_scope('fully_connected') as scope:
                
                self.W_fc1 = self.weight_variable([h_conv3_weights, 512], name='w_fc')
                self.b_fc1 = self.bias_variable([512], name='b_fc')

                # Fully connected layer output
                with tf.name_scope('fully-connected-out') as scope:
                    h_conv3_flat = tf.reshape(self.h_conv3, [tf.negative(1), h_conv3_weights])
                    self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_conv3_flat, self.W_fc1), self.b_fc1))

            # Output layer weights and biases
            with tf.name_scope('output') as scope:
                self.W_fc2 = self.weight_variable([512, self.action_size], name='w_out')
                self.b_fc2 = self.bias_variable([self.action_size], name='b_out')

                # Output
                with tf.name_scope('q_values') as scope:
                    self.q_values = tf.add(tf.matmul(self.h_fc1, self.W_fc2), self.b_fc2)

        if trainable:
            with tf.name_scope('optimizer') as scope:
                self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=0.00025,
                    momentum=0.95,
                    epsilon=0.01)

                with tf.name_scope('loss') as scope:
                    target_q_value = tf.reduce_sum(tf.multiply(self.q_values, self.a), reduction_indices=1)
                    self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y, target_q_value)))
                    #print 'loss: {}'.format(self.loss)

                with tf.name_scope('gradient_clipping') as scope:
                    self.computed_gradients = self.optimizer.compute_gradients(self.loss)   
                    #tf.global_variables()                
                    self.clipped_gradients = [(tf.clip_by_value(g, -1, 1), v) for g,v in self.computed_gradients]

                self.train_op = self.optimizer.apply_gradients(self.clipped_gradients)

    def get_variables(self):
        return [self.W_conv1, self.b_conv1,
                self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3,
                self.W_fc1, self.b_fc1,
                self.W_fc2, self.b_fc2]


    def train(self, s_input, a_input, y_input, trainer_id):
        with tf.device(self.device):
            feed_dict = {
                self.s: s_input,
                self.a: a_input,
                self.y: y_input
                }
            _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            return loss_value


    def predict(self, s_input):
        with tf.device(self.device):
            feed_dict = {self.s: s_input}
            predicted_output = self.sess.run(self.q_values, feed_dict=feed_dict)
            return predicted_output

    def sync_parameters_from(self, source_network):
        with tf.device(self.device):
            source_variables = source_network.get_variables()
            own_variables = self.get_variables()
            sync_ops = []
            for(src_var, own_var) in zip(source_variables, own_variables):            
                sync_op = tf.assign(own_var, src_var)
                sync_ops.append(sync_op)
            return tf.group(*sync_ops)

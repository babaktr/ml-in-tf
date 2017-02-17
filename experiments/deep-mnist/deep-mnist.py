import sys
sys.path.append('../..')

import tensorflow as tf
import numpy as np 
from stats import Stats
from network import ConvolutionalNeuralNetwork

# Get MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

flags = tf.app.flags

flags.DEFINE_integer('minibatches', 20000, 'Number of minibatches to run the training on.')
flags.DEFINE_integer('minibatch_size', 50, 'Number of samples in each minibatch.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate of the optimizer.')
flags.DEFINE_integer('status_update', 100, 'How often to print an status update.')
flags.DEFINE_string('optimizer', 'gradient_descent', 'If another optimizer should be used [adam, rmsprop]. Defaults to gradient descent')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested')
flags.DEFINE_boolean('use_gpu', False, 'If it should run the TensorFlow operations on the GPU rather than the CPU.')

settings = flags.FLAGS

print('Starting session with: Minibatches: {} -- Learning Rate: {} -- Optimizer: {}'.format(settings.minibatches,
                                                                                            settings.learning_rate, 
                                                                                            settings.optimizer)) 

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

cnn_network = ConvolutionalNeuralNetwork(device, settings.random_seed, settings.learning_rate, settings.optimizer)

# Statistics summary writer
summary_dir = '../../logs/deep-mnist-hidden{}-lr{}-minibatch{}-{}/'.format(10, settings.learning_rate, settings.minibatches, settings.optimizer)
summary_writer = tf.summary.FileWriter(summary_dir, cnn_network.sess.graph)
stats = Stats(cnn_network.sess, summary_writer, 2)

for i in range (settings.minibatches): 
    # Get minibatch
    batch = mnist.train.next_batch(settings.minibatch_size)

    # Calculate batch accuracy
    acc = cnn_network.get_accuracy(batch[0], batch[1])

    # Run training
    loss = cnn_network.train(batch[0], batch[1])

    stats.update({'loss': loss, 'accuracy': acc, 'step': i})

    if i % settings.status_update == 0:
        # Print update
        print 'Batch: {}, Loss: {}, Accuracy: {}'.format(i, 
            format(loss, '.4f'), format(acc, '.4f'))

if settings.run_test:   
    print ' '
    print ' --- TESTING MODEL ---'
    print('Accuracy on test set: {}'.format(cnn_network.get_accuracy(mnist.test.images, mnist.test.labels)))
                    

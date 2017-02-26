import sys
sys.path.append('../..')

import tensorflow as tf
import numpy as np 
from stats import Stats
from network import NeuralNetwork

# Get MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

flags = tf.app.flags

flags.DEFINE_integer('minibatches', 1000, 'Number of minibatches to run the training on.')
flags.DEFINE_integer('minibatch_size', 100, 'Number of samples in each minibatch.')
flags.DEFINE_float('learning_rate', 0.5, 'Learning rate of the optimizer.')
flags.DEFINE_integer('status_update', 100, 'How often to print an status update.')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')
flags.DEFINE_string('optimizer', 'gradient_descent', 'Specifices optimizer to use [adam, rmsprop]. Defaults to gradient descent')
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

network = NeuralNetwork(
    device, 
    settings.random_seed, 
    settings.learning_rate, 
    settings.optimizer)

# Statistics summary writer
summary_dir = '../../logs/nn-mnist_minibatches-{}_minibatchsize-{}_lr-{}_optimizer-{}/'.format(
    settings.minibatches, 
    settings.minibatch_size, 
    settings.learning_rate, 
    settings.optimizer)

summary_writer = tf.summary.FileWriter(summary_dir, network.sess.graph)
stats = Stats(network.sess, summary_writer, 2)

for i in range(settings.minibatches): 
    # Get minibatch
    batch_xs, batch_ys = mnist.train.next_batch(settings.minibatch_size)

    # Calculate batch accuracy
    acc = network.get_accuracy(batch_xs, batch_ys)

    # Run training
    loss = network.train(batch_xs, batch_ys)

    stats.update({'loss': loss, 'accuracy': acc, 'step': i})

    if i % settings.status_update == 0:
        # Print update
        print('Minibatches done: {}, Loss: {}, Accuracy: {} %'.format(max(1,i), format(loss, '.4f'), format(acc, '.4f')))

if settings.run_test:
    print(' --- TESTING MODEL ---')
    accuracy = network.get_accuracy(mnist.test.images, mnist.test.labels)
    print('Accuracy on test set: {} %'.format(format(accuracy, '.4f')))


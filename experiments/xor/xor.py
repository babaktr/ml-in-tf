import sys
sys.path.append('../..')

import tensorflow as tf
import numpy as np 
from stats import Stats

from network import NeuralNetwork

flags = tf.app.flags

flags.DEFINE_integer('batches', 10000, 'Number of batches (epochs) to run the training on.')
flags.DEFINE_integer('hidden_n', 2, 'Number of nodes to use in the two hidden layers.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate of the optimizer.')
flags.DEFINE_integer('status_update', 1000, 'How often to print an status update.')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')
flags.DEFINE_string('optimizer', 'gradient_descent', 'Specifices optimizer to use [adam, rmsprop]. Defaults to gradient descent')
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested.')

settings = flags.FLAGS

print('Starting session with: Batches: {} -- Hidden Neurons: {} -- Learning Rate: {} -- Optimizer: {}'.format(settings.batches, 
                                                                                                            settings.hidden_n, 
                                                                                                            settings.learning_rate, 
                                                                                                            settings.optimizer))

# Set up Neural Network
nn_network = NeuralNetwork(settings.random_seed, 
						settings.hidden_n,
						settings.learning_rate,
						settings.optimizer)
# Prepare training data
training_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
training_outputs = [[0.0], [1.0], [1.0], [0.0]]

# Statistics summary writer
summary_dir = '../../logs/xor-hidden{}-lr{}-batches{}-{}/'.format(settings.hidden_n, 
    settings.learning_rate, settings.batches, settings.optimizer)

summary_writer = tf.summary.FileWriter(summary_dir, nn_network.sess.graph)
stats = Stats(nn_network.sess, summary_writer, 2)


for i in range(settings.batches): 
    # Calculate batch accuracy
    acc = nn_network.get_accuracy(training_inputs, training_outputs)

    # Run training
    loss = nn_network.train(training_inputs, training_outputs)

    stats.update({'loss': loss, 'accuracy': acc, 'step': i})
    
    if i % settings.status_update == 0:
        print 'Batch: {}, Loss: {}, Accuracy: {}'.format(i, 
            format(loss, '.4f'), format(acc, '.4f'))

if settings.run_test:
    print ' --- TESTING MODEL ---'
    print('[0.0, 0.0] -- Prediction: {}'.format(nn_network.predict(np.array([[0.0, 0.0]]))))
    print('[0.0, 1.0] -- Prediction: {}'.format(nn_network.predict(np.array([[0.0, 1.0]]))))
    print('[1.0, 0.0] -- Prediction: {}'.format(nn_network.predict(np.array([[1.0, 0.0]]))))
    print('[1.0, 1.0] -- Prediction: {}'.format(nn_network.predict(np.array([[1.0, 1.0]]))))
           

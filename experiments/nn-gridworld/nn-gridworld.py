import sys
sys.path.append('../..')
sys.path.append('../games')

import numpy as np
import tensorflow as tf
from stats import Stats

from network import NeuralNetwork
from gridworld import GridWorld

flags = tf.app.flags

# Q-Learning settings
flags.DEFINE_integer('episodes', 1000, 'Number of episodes to run the training on.')
flags.DEFINE_float('gamma', 0.99, 'Sets the discount in Q-Learning (gamma).')
flags.DEFINE_float('initial_epsilon', 1.0, 'Initial epsilon value that epsilon will be annealed from.')
flags.DEFINE_float('final_epsilon', 0.1, 'Final epsilon value that epsilon will be annealed to.')

# Network settings
flags.DEFINE_integer('hidden_l1', 80, 'Number of neurons in hidden layer 1.')
flags.DEFINE_integer('hidden_l2', 80, 'Number of neurons in hidden layer 2.')

# Training settings
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate of the optimizer.')
flags.DEFINE_string('optimizer', 'rmsprop', 'If another optimizer should be used [adam, gradientdescent, rmsprop]. Defaults to gradient descent.')
flags.DEFINE_integer('train_step_limit', 300, 'Limits the number of steps in training to avoid badly performing agents running forever.')

# General Settings
flags.DEFINE_integer('field_size', 4, 'Determines width and height of the Gridworld field.')
flags.DEFINE_integer('status_update', 10, 'How often to print an status update.')
flags.DEFINE_boolean('use_gpu', False, 'If TensorFlow operations should run on GPU rather than CPU.')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')

# Testing settings
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested.')
flags.DEFINE_integer('test_runs', 100, 'Number of times to run the test.')
flags.DEFINE_float('test_epsilon', 0.1, 'Epsilon to use on test run.')
flags.DEFINE_integer('test_step_limit', 1000, 'Limits the number of steps in test to avoid badly performing agents running forever.')

settings = flags.FLAGS

# Set up GridWorld
env = GridWorld(settings.field_size, settings.random_seed)

np.random.seed(settings.random_seed)

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

input_size = 3 * settings.field_size * settings.field_size

# Set Neural Network
nn_network = NeuralNetwork(device, 
                        settings.random_seed, 
                        input_size,
                        settings.hidden_l1, 
                        settings.hidden_l2, 
                        settings.learning_rate, 
                        settings.optimizer)

# Statistics summary writer
summary_dir = '../../logs/nn-gridworld-fieldsize{}-episodes{}-hidden1_{}-hidden2_{}-lr{}-{}/'.format(settings.field_size,
    settings.episodes, settings.hidden_l1, settings.hidden_l2, settings.learning_rate, settings.optimizer)
summary_writer = tf.summary.FileWriter(summary_dir, nn_network.sess.graph)
stats = Stats(nn_network.sess, summary_writer, 4)

episode = 0
epsilon = settings.initial_epsilon

while settings.episodes > episode:
	# Prepare environment for playing
    state = env.reset()                   

    # Reset or increment values
    terminal = False
    episode += 1
    step = 0
    q_max_arr = []
    reward_arr = []
    epsilon_arr = []
    loss_arr = []
    acc_arr = []

    while not terminal and step < settings.train_step_limit: 
        step += 1
        # Get the Q-values of the current state
        q_values = nn_network.predict(state.reshape(1, input_size))
        # Save max(Q(s,a)) for stats
        q_max = np.max(q_values)
        
        # Anneal epsilon if final epsilon has not been reached
        if epsilon > settings.final_epsilon: 
            epsilon = settings.initial_epsilon - (2*episode / float(settings.episodes))
        else: 
            epsilon = settings.final_epsilon

        # Select random action or action with the highest Q-value
        if (np.random.random() < epsilon): 
            action = np.random.randint(0, 4)
        else: 
            action = np.argmax(q_values)

        # Take action and observe new state and reward, check if state is terminal
        new_state, reward, terminal = env.perform_action(action)
       
       	# Get the new state's Q-values
        q_values_new = nn_network.predict(new_state.reshape(1, input_size))

        # Get max(Q(s',a')) to update Q(s,a)
        q_max_new = np.max(q_values_new)

        # Non-terminal state: update with reward + gamma * max(Q(s',a')
        # Terminal state: update using reward
        if not terminal: 
            update = reward + (settings.gamma * q_max_new)
        else: 
            update = reward

        # Updated the desired output for training the network
        q_values[0][action] = update

        # Calculate accuracy
        acc = nn_network.get_accuracy(state.reshape(1, input_size), q_values) 
                                           
        # Run training
        loss = nn_network.train(state.reshape(1, input_size), q_values)

        # Set the current state to the new state
        state = new_state

        # Save values for stats
        epsilon_arr.append(epsilon)
        reward_arr.append(reward)
        q_max_arr.append(q_max)
        loss_arr.append(loss)
        acc_arr.append(acc)

    # Episode ended, update log and print stats
    stats.update({'loss':np.average(loss_arr), 
                'accuracy': np.average(acc_arr),
                'qmax': np.average(q_max_arr),
                'epsilon': np.average(epsilon_arr),
                'reward': np.sum(reward_arr),
                'steps': step,
                'step': episode
                }) 

    print 'Episode: {}, Steps: {}, Reward: {}, Qmax: {}, Loss: {}, Accuracy: {}, Epsilon: {}'.format(episode, 
        step, np.sum(reward_arr), format(np.average(q_max_arr), '.1f'),  format(np.average(loss_arr), '.4f'), 
        format(np.average(acc_arr), '.2f'), format(np.average(epsilon_arr), '.2f'))

if settings.run_test:
    print ' '
    print ' --- TESTING MODEL ---'
    steps = []
    rewards = []
    for n in range(settings.test_runs):
        state = env.reset()        
        terminal = False
        step = 0
        q_max_arr = []
        reward_arr = []
        epsilon_arr = []

        while not terminal and step < settings.test_step_limit: 
            step += 1

            q_values = nn_network.predict(state.reshape(1, input_size))
            q_max = np.max(q_values)

            if (np.random.random() < settings.test_epsilon): 
                action = np.random.randint(0,4)
            else: 
                action = np.argmax(q_values)

            state, reward, terminal = env.perform_action(action)

            reward_arr.append(reward)
            q_max_arr.append(q_max)
        
            if step == settings.test_step_limit:
                print 'REACHED MAX STEPS'
        
        print 'Run: {}, Steps: {}, Total Reward: {}, Avg Q-max: {}'.format(n, 
        step, format(np.sum(reward_arr), '.1f'), format(np.average(q_max_arr), '.2f'))
        steps.append(step)
        rewards.append(np.average(reward_arr))

    print ' '
    print  ' --- TEST SUMMARY ---'
    print 'Avg steps: {}, Avg reward: {}'.format(np.average(steps), np.average(rewards))
    print 'Max steps: {}, Max rewards: {}'.format(np.max(steps), np.max(rewards))
    print 'Min steps: {}, Min rewards: {}'.format(np.min(steps), np.min(rewards))
    print ' '

import sys
sys.path.append('../..')

import numpy as np
import tensorflow as tf
from stats import Stats
import gym

from network import NeuralNetwork

flags = tf.app.flags

# Q-Learning settings
flags.DEFINE_integer('episodes', 1000, 'Number of episodes to run the training on.')
flags.DEFINE_float('gamma', 0.99, 'Sets the discount in Q-Learning (gamma).')
flags.DEFINE_float('initial_epsilon', 1.0, 'Initial epsilon value that epsilon will be annealed from.')
flags.DEFINE_float('final_epsilon', 0.1, 'Final epsilon value that epsilon will be annealed to.')

# Network settings
flags.DEFINE_integer('hidden_layers', 2, 'Number of hidden layers.')
flags.DEFINE_integer('hidden_nodes', 10, 'Number of neurons in each hidden layer.')
flags.DEFINE_integer('batch_size', 2, 'Size of each training batch.')

# Training settings
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate of the optimizer.')
flags.DEFINE_string('optimizer', 'rmsprop', 'If another optimizer should be used [adam, gradientdescent, rmsprop]. Defaults to rmsprop.')

flags.DEFINE_boolean('use_gpu', False, 'Explanation.')


flags.DEFINE_integer('random_seed', 2, 'Random seed.')
flags.DEFINE_string('game', 'CartPole-v0', 'Classic Control-game to play.')


settings = flags.FLAGS

# Set up GridWorld
env = gym.make(settings.game)

np.random.seed(settings.random_seed)

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

state = env.reset()


print device
print settings.random_seed
print len(state)
print env.action_space.n
print settings.hidden_layers
print settings.hidden_nodes
print settings.learning_rate
print settings.optimizer
# Set Neural Network
nn_network = NeuralNetwork(device, 
                        settings.random_seed, 
                        len(state),
                        env.action_space.n,
                        settings.hidden_layers, 
                        settings.hidden_nodes, 
                        settings.learning_rate, 
                        settings.optimizer)

# Statistics summary writer
summary_dir = '../../logs/nn-classic/'#.format(settings.field_size,
    #settings.episodes, settings.hidden_l1, settings.hidden_l2, settings.learning_rate, settings.optimizer)
summary_writer = tf.summary.FileWriter(summary_dir, nn_network.sess.graph)
stats = Stats(nn_network.sess, summary_writer, 4)

episode = 0
epsilon = settings.initial_epsilon

while settings.episodes > episode:

    # Reset or increment values
    terminal = False
    episode += 1
    step = 0
    q_max_arr = []
    reward_arr = []
    epsilon_arr = []
    loss_arr = []
    acc_arr = []

    state_batch = []
    target_batch = []
    action_batch = []

    state = env.reset()

    while not terminal: 
        step += 1
        # Get the Q-values of the current state
        q_values = nn_network.predict(state)
        # Save max(Q(s,a)) for stats
        q_max = np.max(q_values)
        env.render()
        
        # Anneal epsilon if final epsilon has not been reached
        if epsilon > settings.final_epsilon: 
            epsilon = settings.initial_epsilon - (2*episode / float(settings.episodes))
        else: 
            epsilon = settings.final_epsilon

        # Select random action or action with the highest Q-value
        if (np.random.random() < epsilon): 
            action = np.random.randint(0, env.action_space.n)
        else: 
            action = np.argmax(q_values)

        # Take action and observe new state and reward, check if state is terminal
        new_state, reward, terminal, _ = env.perform_action(action)
       
       	# Get the new state's Q-values
        q_values_new = nn_network.predict(new_state)

        # Get max(Q(s',a')) to update Q(s,a)
        q_max_new = np.max(q_values_new)

        # Non-terminal state: update with reward + gamma * max(Q(s',a')
        # Terminal state: update using reward
        if not terminal: 
            update = reward + (settings.gamma * q_max_new)
        else: 
            update = reward

        onehot_action = np.zeros(env.action_space.n)
        onehot_action[action] = 1

        state_batch.append(state)
        target_batch.append([update])
        action_batch.append(onehot_action)

        # Save values for stats
        epsilon_arr.append(epsilon)
        reward_arr.append(reward)
        q_max_arr.append(q_max)
        loss_arr.append(loss)
        acc_arr.append(acc)


        # Calculate accuracy
        #acc = nn_network.get_accuracy(state.reshape(1, input_size), q_values) 
        if step % settings.batch_size == 0 or terminal:
            # Run training
            loss = nn_network.train(state_batch, action_batch, target_batch, settings.learning_rate)

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
        else:
            # Set the current state to the new state
            state = new_state

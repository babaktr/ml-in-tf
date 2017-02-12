import sys
sys.path.append('../..')

import numpy as np
import tensorflow as tf
from stats import Stats
import random
import gym

from network import NeuralNetwork
from experience_replay import ExperienceReplayMemory

flags = tf.app.flags

# Q-Learning settings
flags.DEFINE_integer('episodes', 500, 'Number of episodes to run the training on.')
flags.DEFINE_float('gamma', 0.99, 'Sets the discount in Q-Learning (gamma).')
flags.DEFINE_float('initial_epsilon', 0.4, 'Initial epsilon value that epsilon will be annealed from.')
flags.DEFINE_float('final_epsilon', 0.01, 'Final epsilon value that epsilon will be annealed to.')

# Network settings
flags.DEFINE_integer('hidden_layers', 2, 'Number of hidden layers.')
flags.DEFINE_integer('hidden_nodes', 5, 'Number of neurons in each hidden layer.')
flags.DEFINE_integer('batch_size', 64, 'Size of each training batch.')
flags.DEFINE_integer('replay_size', 100000, 'Size of each training batch.')


# Training settings
flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate of the optimizer.')
flags.DEFINE_float('final_learning_rate', 0.0001, 'Learning rate of the optimizer.')
flags.DEFINE_boolean('lr_anneal', True, 'lr anneal.')


flags.DEFINE_string('optimizer', 'rmsprop', 'If another optimizer should be used [adam, gradientdescent, rmsprop]. Defaults to rmsprop.')

flags.DEFINE_boolean('use_gpu', False, 'Explanation.')



flags.DEFINE_integer('random_seed', 1, 'Random seed.')
flags.DEFINE_string('game', 'CartPole-v0', 'Classic Control-game to play.')


settings = flags.FLAGS

# Set up GridWorld
env = gym.make(settings.game)

np.random.seed(settings.random_seed)
random.seed(settings.random_seed)

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

state = env.reset()

# Set Neural Network
nn_network = NeuralNetwork(device, 
                        settings.random_seed, 
                        len(state),
                        env.action_space.n,
                        settings.hidden_layers, 
                        settings.hidden_nodes, 
                        settings.learning_rate, 
                        settings.optimizer)

memory = ExperienceReplayMemory(settings.replay_size)

# Statistics summary writer
summary_dir = '../../logs/slowannealeps-NEGREW-nn-classic_game-{}_episodes-{}_hiddenlayers-{}_hiddennodes-{}_batchsize-{}_replaysize-{}_lr-{}_lranneal-{}_finallr-{}_optimizer-{}_gamma{}/'.format(settings.game, settings.episodes,
    settings.hidden_layers, settings.hidden_nodes, settings.batch_size, settings.replay_size, settings.learning_rate, settings.lr_anneal, settings.final_learning_rate, settings.optimizer, settings.gamma)
summary_writer = tf.summary.FileWriter(summary_dir, nn_network.sess.graph)
stats = Stats(nn_network.sess, summary_writer, 4)

episode = 0
epsilon = settings.initial_epsilon
lr = settings.learning_rate

state_replay = []
target_replay = []
action_replay = []

# fill replay memory
print 'fill replay memory'
replay_episodes = 0
for n in range(5000):
    if replay_episodes % 100 == 0:
        print 'replay_episode: {}, replay length: {}, progress: {}'.format(replay_episodes, len(state_replay), len(state_replay)/float(30000))
    terminal = False
    state = env.reset()
    replay_episodes += 1
    while not terminal: 
        # Get the Q-values of the current state
        #q_values = nn_network.predict([state])
        # Save max(Q(s,a)) for stats
        #q_max = np.max(q_values)
        #env.render()
       
        action = np.random.randint(0, env.action_space.n)

        # Take action and observe new state and reward, check if state is terminal
        new_state, reward, terminal, _ = env.step(action)

        if terminal:
            reward = -1.0

        onehot_action = np.zeros(env.action_space.n)
        onehot_action[action] = 1.0

        memory.save(state, onehot_action, reward, new_state, terminal)

        state = new_state

print 'start playing'
global_step = 0 
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
    #new_state_batch = []
    target_batch = []
    action_batch = []

    

    final_lr = 0.000001
    state = env.reset()

    if settings.lr_anneal:
        lr = settings.learning_rate - episode*((settings.learning_rate-settings.final_learning_rate)/float(settings.episodes))
    else:
        lr = settings.learning_rate

    while not terminal and step < 200: 
        step += 1
        global_step += 1
        # Get the Q-values of the current state
        q_values = nn_network.predict([state])
        # Save max(Q(s,a)) for stats
        q_max = np.max(q_values)
        env.render()
        
        # Anneal epsilon if final epsilon has not been reached
        if epsilon > settings.final_epsilon: 
            epsilon = settings.initial_epsilon - (2*episode / float(settings.episodes))
        else: 
            epsilon = settings.final_epsilon


        # Select random action or action with the highest Q-value
        if np.random.random() < epsilon: 
            action = np.random.randint(0, env.action_space.n)
        else: 
            action = np.argmax(q_values)

        # Take action and observe new state and reward, check if state is terminal
        new_state, reward, terminal, _ = env.step(action)
        #env.render()
        onehot_action = np.zeros(env.action_space.n)
        onehot_action[action] = 1.0

        if terminal:
            reward = -1.0

        # Save values for stats
        epsilon_arr.append(epsilon)
        reward_arr.append(reward)
        q_max_arr.append(q_max)
        acc_arr.append(0)

        memory.save(state, onehot_action, reward, new_state, terminal)

        # Non-terminal state: update with reward + gamma * max(Q(s',a')
        # Terminal state: update using reward
        sample_s_t, sample_a_t, sample_r_t, sample_s_t1, sample_terminal = memory.sample(settings.batch_size)

        for n in range(len(sample_s_t)):
            s_t = sample_s_t[n]
            a_t = sample_a_t[n]
            r_t = sample_r_t[n]
            s_t1 = sample_s_t1[n]
            term = sample_terminal[n]

            q_t1 = nn_network.predict([s_t1])

            if not term: 
                target = r_t + (settings.gamma * np.argmax(q_t1))
            else:
                target = r_t

            state_batch.append([s_t])
            action_batch.append(a_t)
            target_batch.append([target])
       
        # Run training
        loss = nn_network.train(state_batch, action_batch, target_batch, lr)
        state_batch = []
        action_batch = []
        target_batch = []
        
        loss_arr.append(loss)

        if terminal:
            # Episode ended, update log and print stats
            stats.update({'loss':np.average(loss_arr), 
                    'accuracy': np.average(acc_arr),
                    'qmax': np.average(q_max_arr),
                    'epsilon': np.average(epsilon_arr),
                    'reward': np.sum(reward_arr),
                    'steps': step,
                    'step': episode
                    }) 
            print 'Episode: {}, Global steps: {}, Steps: {}, Reward: {}, Qmax: {}, Loss: {}, lr: {}, Epsilon: {}'.format(episode, global_step,
                    step, np.sum(reward_arr), format(np.average(q_max_arr), '.1f'),  format(np.average(loss_arr), '.4f'), 
                    format(lr, '.6f'), format(np.average(epsilon_arr), '.3f'))

        else:
            # Set the current state to the new state
            state = new_state

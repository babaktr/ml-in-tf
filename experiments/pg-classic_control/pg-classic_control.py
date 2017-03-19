import sys
sys.path.append('../..')
sys.path.append('../games')

import gym
import random
import numpy as np
import tensorflow as tf
from stats import Stats

from network import NeuralNetworks

flags = tf.app.flags

# Q-Learning settings
flags.DEFINE_integer('episodes', 2000, 'Number of episodes to run the training on.')
flags.DEFINE_float('gamma', 0.99, 'Sets the discount in Q-Learning (gamma).')

# Network settings
flags.DEFINE_integer('hidden_size', 10, 'Number of neurons in the hidden layer.')

# Training settings
flags.DEFINE_float('policy_learning_rate', 0.001, 'Learning rate of the optimizer.')
flags.DEFINE_float('value_learning_rate', 0.01, 'Learning rate of the optimizer.')


flags.DEFINE_string('optimizer', 'adam', 'If another optimizer should be used [adam, gradientdescent, rmsprop]. Defaults to gradient descent.')
flags.DEFINE_integer('train_step_limit', 300, 'Limits the number of steps in training to avoid badly performing agents running forever.')


# General Settings
flags.DEFINE_string('game', 'CartPole-v0', 'The game to play.')
flags.DEFINE_integer('status_update', 10, 'How often to print an status update.')
flags.DEFINE_boolean('use_gpu', False, 'If TensorFlow operations should run on GPU rather than CPU.')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')

# Testing settings
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested.')
flags.DEFINE_integer('test_runs', 100, 'Number of times to run the test.')
flags.DEFINE_float('test_epsilon', 0.1, 'Epsilon to use on test run.')
flags.DEFINE_integer('test_step_limit', 1000, 'Limits the number of steps in test to avoid badly performing agents running forever.')

settings = flags.FLAGS

def run_episode(env, networks):
    state = env.reset()
    total_reward = 0
    step = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    target_values = []
    q_max_arr = []
    value_arr = []

    p_loss = 0
    v_loss = 0

    for n in range(200):
        # Compute policy
        policy = networks.predict_policy(state)[0]
        q_max_arr.append(np.max(policy))
        action = np.random.choice(env.action_space.n, 1, p=policy)[0]
        # Record
        states.append(state)
        onehot_action = np.eye(1, env.action_space.n, action)[0]
        #print 'onehot: {}'.format(onehot_action)
        actions.append(onehot_action)

        new_state, reward, terminal, _ = env.step(action)

        transitions.append((state, action, reward))
        total_reward += reward
        step += 1

        if terminal:
            break
        else:
            state = new_state

    for index, transition in enumerate(transitions):
        state, action, reward = transition

        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1

        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97

        value = networks.predict_value(state)[0][0]
        value_arr.append(value)
        #print 'value: {}'.format(value)
        advantages.append(future_reward - value)

        target_values.append(future_reward)

    # Train value network
    total_v_loss = networks.train_value(states, target_values)
    avg_v_loss = total_v_loss/len(transitions)

    # Train policy network
    total_p_loss = networks.train_policy(states, actions, advantages)
    avg_p_loss = total_p_loss/len(transitions)


    return total_reward, avg_v_loss, avg_p_loss, step, np.average(q_max_arr), np.average(value_arr)

# Set up Classic Control Enviroment
env = gym.make(settings.game)
state = env.reset()

np.random.seed(settings.random_seed)

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

# Set Neural Network
networks = NeuralNetworks(device, 
                        settings.random_seed, 
                        len(state),
                        env.action_space.n,
                        settings.hidden_size,
                        settings.policy_learning_rate, #p lr
                        settings.value_learning_rate, #v lr
                        settings.optimizer)

# Statistics summary writer
summary_dir = '../../logs/pg-classic_control-episodes{}-hidden_{}-plr{}-vlr_{}-{}/'.format(settings.episodes, 
     settings.hidden_size, settings.policy_learning_rate, settings.value_learning_rate, settings.optimizer)
summary_writer = tf.summary.FileWriter(summary_dir, networks.sess.graph)
stats = Stats(networks.sess, summary_writer, 5)

for i in range(settings.episodes):
    rewards, v_loss, p_loss, step, avg_qmax, avg_value = run_episode(env, networks)
    print('Episode: {}, Reward: {}'.format(i, rewards))

    stats.update({'policy_loss': p_loss, 
                'value_loss': v_loss,
                'avg_qmax': avg_qmax,
                'avg_value': avg_value,
                'reward': rewards,
                'steps': step,
                'step': i
                }) 

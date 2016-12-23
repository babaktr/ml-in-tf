import sys
sys.path.append('../..')
sys.path.append('../games')

import numpy as np
import tensorflow as tf
from stats import Stats

import random
from gridworld import GridWorld

flags = tf.app.flags

# Q Learning settings
flags.DEFINE_integer('episodes', 100, 'Number of minibatches to run the training on.')
flags.DEFINE_float('gamma', 0.99, 'Discount to use when Q-value is updated.')
flags.DEFINE_float('initial_epsilon', 1.0, 'Initial epsilon value that epsilon will be annealed from.')
flags.DEFINE_float('final_epsilon', 0.1, 'Final epsilon value that epsilon will be annealed to.')

# Training settings
flags.DEFINE_float('learning_rate', 0.5, 'Learning rate of the optimizer.')

# General settings
flags.DEFINE_float('test_epsilon', 0.1, 'Epsilon to use on test run.')
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested')
flags.DEFINE_integer('random_seed', 123, 'Number of minibatches to run the training on.')

settings = flags.FLAGS                                      

env = GridWorld(settings.random_seed)
sess = tf.InteractiveSession()

episode = 0
epsilon = settings.initial_epsilon

summary_dir = '../../logs/q-gridworld-episodes{}-lr{}/'.format(settings.episodes, 
    settings.learning_rate)
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
stats = Stats(sess, summary_writer, 3)

while settings.episodes > episode:
    # Prepare environment for playing
    env.reset()                   

    # Reset or increment values
    terminal = False
    episode += 1
    step = 0
    q_max_arr = []
    reward_arr = []
    epsilon_arr = []

    while not terminal:
        step += 1
        # Get the Q-values of the current state
        q_values = env.q_values()
        # Save max(Q(s,a)) for stats
        q_max = np.max(q_values)

        # Anneal epsilon
        if epsilon > settings.final_epsilon: 
            epsilon = settings.initial_epsilon - (2*episode / float(settings.episodes))
        else: 
            # Final epsilon reached, stop annealing.
            epsilon = settings.final_epsilon

        # Select action
        if (random.random() < epsilon): 
            # Choose random action
            action = np.random.randint(0,4)
        else: 
            # Choose the action with the highest Q-value
            action = np.argmax(q_values)

        # Take action and observe reward and check if state is terminal
        _, reward, terminal = env.perform_action(action)

        # Save values for stats
        epsilon_arr.append(epsilon)
        reward_arr.append(reward)
        q_max_arr.append(q_max)
        
        # Get the new states Q-values
        q_values_new = env.q_values()
        # Get max(Q(s',a')) to update Q(s,a)
        q_max_new = np.max(q_values_new)
        
        if not terminal: 
            # Non-terminal state, update with reward + gamma * max(Q(s'a')
            update = reward + (settings.gamma * q_max_new)
        else: 
            # Terminal state, update using reward
            update = reward

        # Update Q-table
        env.update_q_table(update, action, settings.learning_rate, terminal)

    stats.update({'qmax': np.average(q_max_arr),
                'epsilon': np.average(epsilon_arr),
                'reward': np.sum(reward_arr),
                'steps': step,
                'step': episode
                })    
    print 'Episode: {}, Steps: {}, Total Reward: {}, Avg Q-max: {}, Avg Epsilon: {}'.format(episode, 
        step, format(np.sum(reward_arr), '.1f'), format(np.average(q_max_arr), '.2f'),
        format(np.average(epsilon_arr), '.2f'))



if settings.run_test:
    print ' '
    print ' --- TESTING MODEL ---'
    steps = []
    rewards = []

    for n in range(100):
        env.reset()        
        terminal = False
        step = 0
        q_max_arr = []
        reward_arr = []
        epsilon_arr = []

        while not terminal:
            step += 1
            
            q_values = env.q_values()
            q_max = np.max(q_values)

            if (random.random() < settings.test_epsilon): 
                action = np.random.randint(0,4)
            else: 
                action = np.argmax(q_values)

            _, reward, terminal = env.perform_action(action)

            reward_arr.append(reward)
            q_max_arr.append(q_max)
            
        print 'Run: {}, Steps: {}, Total Reward: {}, Avg Q-max: {}'.format(n, 
            step, format(np.sum(reward_arr), '.1f'), format(np.average(q_max_arr), '.2f'))

        steps.append(step)
        rewards.append(np.average(reward_arr))

    print ' '
    print  ' --- TEST SUMMARY ---'
    print 'Avg steps: {}, Avg reward: {}'.format(np.average(steps), np.average(rewards))
    print 'Max steps: {}, Max rewards: {}'.format(np.max(steps), np.max(rewards))
    print 'Min steps: {}, Min rewards: {}'.format(np.min(steps), np.min(rewards))
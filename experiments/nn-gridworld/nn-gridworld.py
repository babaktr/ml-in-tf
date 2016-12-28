import sys
sys.path.append('../..')
sys.path.append('../games')

import numpy as np
import tensorflow as tf
from stats import Stats

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
flags.DEFINE_boolean('use_gpu', False, 'If it should run on GPU rather than CPU.')
flags.DEFINE_integer('random_seed', 123, 'Sets the random seed.')

# Testing settings
flags.DEFINE_boolean('run_test', True, 'If the final model should be tested.')
flags.DEFINE_integer('test_runs', 100, 'Number of times to run the test.')
flags.DEFINE_float('test_epsilon', 0.1, 'Epsilon to use on test run.')
flags.DEFINE_integer('test_step_limit', 1000, 'Limits the number of steps in test to avoid badly performing agents running forever.')

settings = flags.FLAGS

env = GridWorld(settings.field_size, settings.random_seed)
sess = tf.InteractiveSession()
np.random.seed(settings.random_seed)

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

input_size = 3 * settings.field_size * settings.field_size

with tf.device(device):
    # Input with shape [?, input_size]
    x = tf.placeholder(tf.float32, shape=[None, input_size], name='x-input')
    # Desired output with shape [?, 4]
    y_ = tf.placeholder(tf.float32, shape=[None, 4], name='desired-output')

    # Hidden layer 1 weights and bias
    W_1 = tf.Variable(tf.random_uniform([input_size, settings.hidden_l1]), name='weights-1')
    b_1 = tf.Variable(tf.zeros([settings.hidden_l1]), name='bias-1')

    # Hidden layer 1's output
    with tf.name_scope('hidden-layer-1') as scope:
        out_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

    # Hidden layer 2 weights and bias 
    W_2 = tf.Variable(tf.random_uniform([settings.hidden_l1, settings.hidden_l2]), name='weights-2')
    b_2 = tf.Variable(tf.zeros([settings.hidden_l2]), name='bias-2')

    # Hidden layer 2's output 
    with tf.name_scope('hidden-layer-2') as scope:
        out_2 = tf.nn.relu(tf.matmul(out_1, W_2) + b_2)

    # Hidden layer 3 weights and bias 
    W_3 = tf.Variable(tf.random_uniform([settings.hidden_l2, 4]), name='weights-3')
    b_3 = tf.Variable(tf.zeros([4]), name='bias-3')

    # Hidden layer 3's output 
    with tf.name_scope('output') as scope:
        y = tf.matmul(out_2, W_3) + b_3

    # Objective function 
    with tf.name_scope('loss') as scope:
        obj_function = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y_, y))))

    with tf.name_scope('train') as scope:
        if settings.optimizer.lower() == 'adam':
            # Adam Optimizer
            train_step = tf.train.AdamOptimizer(settings.learning_rate).minimize(obj_function)
        elif settings.optimizer.lower() == 'rmsprop':
            # RMSProp
            train_step = tf.train.RMSPropOptimizer(settings.learning_rate).minimize(obj_function)
        else: 
            # Gradient Descent
            train_step = tf.train.GradientDescentOptimizer(settings.learning_rate).minimize(obj_function)


    init = tf.global_variables_initializer()
    sess.run(init)

# Specify how accuracy is measured
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Statistics summary writer
summary_dir = '../../logs/nn-gridworld-fieldsize{}-episodes{}-hidden1_{}-hidden2_{}-lr{}-{}/'.format(settings.field_size,
    settings.episodes, settings.hidden_l1, settings.hidden_l2, settings.learning_rate, settings.optimizer)
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
stats = Stats(sess, summary_writer, 4)

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
        q_values = sess.run(y, feed_dict={x: state.reshape(1, input_size)})
        # Save max(Q(s,a)) for stats
        q_max = np.max(q_values)
        
        # Anneal epsilon
        if epsilon > settings.final_epsilon: 
            epsilon = settings.initial_epsilon - (2*episode / float(settings.episodes))
        else: 
            # Final epsilon reached, stop annealing.
            epsilon = settings.final_epsilon

        # Select action
        if (np.random.random() < epsilon): 
            # Choose random action
            action = np.random.randint(0, 4)
        else: 
            # Choose the action with the highest Q-value
            action = np.argmax(q_values)

        # Take action and observe new state and reward, check if state is terminal
        new_state, reward, terminal = env.perform_action(action)
       
       	# Get the new state's Q-values
        q_values_new = sess.run(y, feed_dict={x: new_state.reshape(1, input_size)})
        # Get max(Q(s',a')) to update Q(s,a)
        q_max_new = np.max(q_values_new)

        if not terminal: 
            # Non-terminal state, update with reward + gamma * max(Q(s'a')
            update = reward + (settings.gamma * q_max_new)
        else: 
            # Terminal state, update using reward
            update = reward

        # Updated the desired output for training the network
        q_values[0][action] = update

        # Run training
        _, loss = sess.run([train_step, obj_function],
                    feed_dict={x: state.reshape(1, input_size),
                            y_: q_values})

        # Calculate accuracy
        acc = sess.run(accuracy, feed_dict={x: state.reshape(1, input_size), 
                                            y_: q_values})

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
    if episode % settings.status_update == 0:
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

            q_values = sess.run(y, feed_dict={x: state.reshape(1, input_size)})
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

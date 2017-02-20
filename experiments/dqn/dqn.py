import sys
sys.path.append('../..')

import numpy as np
import signal
import tensorflow as tf
from stats import Stats
import random
import gym
import time
import os

from game_state import GameState

from network import DeepQNetwork
from experience_replay import ExperienceReplayMemory

flags = tf.app.flags

# Q-Learning settings
flags.DEFINE_float('gamma', 0.99, 'Sets the discount in Q-Learning (gamma).')
flags.DEFINE_float('initial_epsilon', 1.0, 'Initial epsilon value that epsilon will be annealed from.')
flags.DEFINE_float('final_epsilon', 0.1, 'Final epsilon value that epsilon will be annealed to.')
flags.DEFINE_float('epsilon_anneal_steps', 1000000, 'Final epsilon value that epsilon will be annealed to.')


# Network settings
flags.DEFINE_integer('batch_size', 32, 'Size of each training batch.')
flags.DEFINE_integer('max_step', 10000000, '')
flags.DEFINE_integer('target_update', 10000, '')
flags.DEFINE_float('gradient_clip_norm', 40., 'Size of each training batch.')
flags.DEFINE_integer('experience_replay_size', 1000000, '')

flags.DEFINE_integer('no_op_max', 30, 'h')



flags.DEFINE_integer('evaluation_frequency', 100000, '')
flags.DEFINE_boolean('run_evaluation', True, '')



# Training settings
flags.DEFINE_float('initial_learning_rate', 0.0007, 'Learning rate of the optimizer.')
flags.DEFINE_float('final_learning_rate', 0., 'Learning rate of the optimizer.')

flags.DEFINE_string('optimizer', 'rmsprop', 'If another optimizer should be used [adam, gradientdescent, rmsprop]. Defaults to rmsprop.')

flags.DEFINE_boolean('use_gpu', False, 'Explanation.')
flags.DEFINE_boolean('display', False, 'Explanation.')



flags.DEFINE_integer('random_seed', 1, 'Random seed.')
flags.DEFINE_string('game', 'BreakoutDeterministic-v0', 'Classic Control-game to play.')


settings = flags.FLAGS

def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True

def load_checkpoint(sess, saver, checkpoint_path):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print('Checkpoint loaded:', checkpoint.model_checkpoint_path)
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_step = int(tokens[len(tokens)-1])
        print('Global step set to: ', global_step)
        # set wall time
        wall_t_fname = checkpoint_path + '/' + 'wall_t.' + str(global_step)
        with open(wall_t_fname, 'r') as f:
            wall_t = float(f.read())
    else:
        print('Could not find old checkpoint')
        global_step = 0
        wall_t = 0.0
    return wall_t, global_step


def init_checkpoint():
    checkpoint_dir = './checkpoints/{}/'.format(settings.game)
    saver = tf.train.Saver(max_to_keep=1)
    wall_t, total_step = load_checkpoint(sess, saver, checkpoint_dir)
    return wall_t, total_step, saver, checkpoint_dir

def save_checkpoint(sess, saver, checkpoint_dir, wall_t, total_step, start_time):
    print('Now saving checkpoint. Please wait.')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')  
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)  

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = checkpoint_dir + '/' + 'wall_t.' + str(total_step)
    with open(wall_t_fname, 'w') as f:
        f.write(str(wall_t))

    saver.save(sess, checkpoint_dir + '/' 'checkpoint', global_step=total_step)

def push_stats_updates(stats, loss_arr, learning_rate, q_max_arr, epsilon_arr, action_arr, reward_arr, l_step, g_step):
    stats.update({'loss': np.average(loss_arr), 
                'learning_rate': learning_rate,
                'qmax': np.average(q_max_arr),
                'epsilon': np.average(epsilon_arr),
                'episode_actions': action_arr,
                'reward': np.sum(reward_arr),
                'steps': l_step,
                'step': g_step
                }) 

def init_networks():
    online = DeepQNetwork(
        sess,
        device,
        'online_network',
        settings.random_seed, 
        game_state.action_size, 
        trainable=True,
        optimizer=settings.optimizer,
        gradient_clip_norm=settings.gradient_clip_norm)

    target = DeepQNetwork(
        sess,
        device,
        'target_network',
        settings.random_seed, 
        game_state.action_size)

    return online, target

def anneal_learning_rate(initial_rate, final_rate, current_step, anneal_steps):
    return initial_rate - current_step * ((initial_rate - final_rate) / float(anneal_steps))

def anneal_epsilon(initial_epsilon, final_epsilon, current_step, anneal_steps):
    return initial_epsilon - current_step * ((initial_epsilon - final_epsilon) / float(anneal_steps))

def select_action(q_values, epsilon, action_size):
    if np.random.random() < epsilon: 
        return np.random.randint(0, action_size)
    else: 
        return np.argmax(q_values)

def train_agent(total_step, stats):
    episode = 0
    epsilon = settings.initial_epsilon

    while settings.max_step > total_step and not stop_requested:

        # Reset or increment values
        terminal = False
        episode += 1
        step = 0
        q_max_arr = []
        reward_arr = []
        epsilon_arr = []
        action_arr = []
        loss_arr = []
        acc_arr = []

        learning_rate = anneal_learning_rate(
            settings.initial_learning_rate, 
            settings.final_learning_rate, 
            total_step, 
            settings.max_step)

        epsilon = anneal_epsilon(
            settings.initial_epsilon, 
            settings.final_epsilon, 
            total_step,
            settings.epsilon_anneal_steps)

        state, reward, terminal = game_state.reset()

        while not terminal and not stop_requested: 
            step += 1
            total_step += 1
            #print(total_step)
            # Get the Q-values of the current state
            q_values = online_network.predict([state])
            #print(q_values)
            # Save max(Q(s,a)) for stats
            q_max = np.max(q_values)

            action = select_action(q_values, epsilon, game_state.action_size)

            # Take action and observe new state and reward, check if state is terminal
            new_state, reward, terminal = game_state.step(action)

            onehot_action = np.zeros(game_state.action_size)
            onehot_action[action] = 1.0

            # Save values for stats
            epsilon_arr.append(epsilon)
            reward_arr.append(reward)
            q_max_arr.append(q_max)
            acc_arr.append(0)
            action_arr.append(action)

            if terminal:
                new_state = None

            memory.save(state, onehot_action, reward, new_state, terminal)

            sample_s_t, sample_a_t, sample_r_t, sample_s_t1, sample_terminal = memory.sample(settings.batch_size)
            state_batch, action_batch, target_batch = [], [], []

            for n in range(len(sample_s_t)):
                s_t = sample_s_t[n]
                a_t = sample_a_t[n]
                r_t = sample_r_t[n]
                s_t1 = sample_s_t1[n]
                term = sample_terminal[n]

                if not term: 
                    q_t1 = target_network.predict([s_t1])
                    target = r_t + (settings.gamma * np.max(q_t1))
                else:
                    target = r_t

                state_batch.append([s_t])
                action_batch.append(a_t)
                target_batch.append([target])
                
            loss = online_network.train(state_batch, action_batch, target_batch, learning_rate)
            state_batch, action_batch, target_batch = [], [], []
            
            loss_arr.append(loss)

            if total_step % settings.target_update == 0:
                sess.run(target_network.sync_parameters_from(online_network))

            if terminal:
                # Episode ended, update log and print stats
                print('Episode: {}, Total steps: {}, Steps: {}, Reward: {}, Qmax: {}, Loss: {}, lr: {}, Epsilon: {}'.format(episode, total_step,
                        step, np.sum(reward_arr), format(np.average(q_max_arr), '.2f'),  format(np.average(loss_arr), '.4f'), 
                        format(learning_rate, '.6f'), format(np.average(epsilon_arr), '.3f')))
                push_stats_updates(stats, loss_arr, learning_rate, q_max_arr, epsilon_arr, action_arr, reward_arr, step, total_step)


            else:
                # Set the current state to the new state
                state = new_state
    return total_step


signal.signal(signal.SIGINT, signal_handler)
stop_requested = False
wall_t = 0
total_step = 0

# Set up GridWorld
game_state = GameState(settings.random_seed,
    settings.game,
    settings.display,
    settings.no_op_max)

np.random.seed(settings.random_seed)
random.seed(settings.random_seed)

memory = ExperienceReplayMemory(settings.experience_replay_size)

if settings.use_gpu:
    device = '/gpu:0'
else:
    device = '/cpu:0'

with tf.device(device):
    sess = tf.Session(
        config=tf.ConfigProto(
           log_device_placement=False, 
           allow_soft_placement=True))

    online_network, target_network = init_networks()

    summary_dir = './logs/{}/'.format(settings.game)
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    stats = Stats(sess, summary_writer, 50)

    wall_t, total_step, saver, checkpoint_dir = init_checkpoint()

    init = tf.global_variables_initializer()
    sess.run(init)

start_time = time.time() - wall_t

print('start training')
total_step = train_agent(total_step, stats)

save_checkpoint(sess, saver, checkpoint_dir, wall_t, total_step, start_time)

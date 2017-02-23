import sys
sys.path.append('../..')

import numpy as np
import signal
import tensorflow as tf
from stats2 import Stats
import random
import gym
import time
import os

from multiprocessing import Queue

from game_state import GameState

from network import DeepQNetwork
from agent import Agent
from experience_replay import ExperienceReplayMemory

from predictor_online import PredictorOnline
from predictor_target import PredictorTarget

from trainer import Trainer


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

flags.DEFINE_integer('predict_batch_size', 128, ' ')

flags.DEFINE_integer('evaluation_frequency', 100000, '')
flags.DEFINE_boolean('run_evaluation', True, '')

flags.DEFINE_boolean('play_mode', False, '')


flags.DEFINE_integer('max_queue_size', 200, '')




# Training settings
flags.DEFINE_float('initial_learning_rate', 0.0007, 'Learning rate of the optimizer.')
flags.DEFINE_float('final_learning_rate', 0., 'Learning rate of the optimizer.')

flags.DEFINE_string('optimizer', 'rmsprop', 'If another optimizer should be used [adam, gradientdescent, rmsprop]. Defaults to rmsprop.')

flags.DEFINE_boolean('use_gpu', False, 'Explanation.')
flags.DEFINE_boolean('display', False, 'Explanation.')



flags.DEFINE_integer('random_seed', 1, 'Random seed.')
flags.DEFINE_string('game', 'BreakoutDeterministic-v0', 'Classic Control-game to play.')

args = flags.FLAGS

class main:
    def __init__(self, settings):
        signal.signal(signal.SIGINT, self.signal_handler)
        self.stop_requested = False
        wall_t = 0
        total_step = 0

        np.random.seed(settings.random_seed)
        random.seed(settings.random_seed)
        self.gamma = settings.gamma
        self.batch_size = settings.batch_size

        self.experience_replay = ExperienceReplayMemory(settings.experience_replay_size)

        self.predictor_online = PredictorOnline(self, settings.predict_batch_size)
        self.predictor_target = PredictorTarget(self, settings.predict_batch_size)
        self.trainers = [
        	Trainer(self, 0),
        	Trainer(self, 1),
        	Trainer(self, 2),
        	Trainer(self, 3),
        	Trainer(self, 4)]


        self.prediction_queue = Queue(maxsize=settings.max_queue_size)
        self.target_prediction_queue = Queue(maxsize=settings.max_queue_size)
        self.training_queue = Queue(maxsize=settings.max_queue_size)


        self.stats = Stats(total_step)

        epsilon_settings = {'initial_epsilon': settings.initial_epsilon, 
            'final_epsilon': settings.final_epsilon, 
            'anneal_steps': settings.epsilon_anneal_steps}

        self.agent = Agent(
            self.prediction_queue, 
            self.target_prediction_queue, 
            self.training_queue, 
            self.stats.episode_log_queue, 
            self.experience_replay, 
            epsilon_settings,
            random_seed=settings.random_seed, 
            game=settings.game, 
            display=settings.display, 
            no_op_max=settings.no_op_max, 
            play_mode=settings.play_mode,
            batch_size=settings.batch_size)
            
        if settings.use_gpu:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        print('device selected')

        with tf.device(device):
            self.sess = tf.Session(
                config=tf.ConfigProto(
                   log_device_placement=False, 
                   allow_soft_placement=True))

            self.online_network, self.target_network = self.init_networks(self.sess, device, settings.random_seed, self.agent.game_state.action_size, settings.gradient_clip_norm)

            summary_dir = './logs/{}/'.format(settings.game)
            summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
            #stats = Stats(sess, summary_writer, 50)
            

            wall_t, total_step, saver, checkpoint_dir = self.init_checkpoint(self.sess, settings.game)

            init = tf.global_variables_initializer()
            self.sess.run(init)

        start_time = time.time() - wall_t

        print('Start training')
        time.sleep(1)
        self.predictor_target.start()
        self.predictor_online.start()
        for t in self.trainers:
        	t.start()
        self.agent.start()
        self.stats.start()


        while self.stats.total_steps.value < settings.max_step:
            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if self.stop_requested:
                #self.save_checkpoint(self.sess, saver, checkpoint_dir, wall_t, self.stats.total_steps.value, start_time)
                self.agent.stop_flag.value = 1
                self.predictor_online.stop_flag = True
                self.predictor_target.stop_flag = True
                self.trainer.stop_flag = True
                self.stats.stop_flag = True
                break

            time.sleep(0.1)
        print('Ending?')





    def signal_handler(self, signal, frame):
        print('You pressed Ctrl+C!')
        self.stop_requested = True

    def load_checkpoint(self, sess, saver, checkpoint_path):
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

    def init_checkpoint(self, sess, game):
        checkpoint_dir = './checkpoints/{}/'.format(game)
        saver = tf.train.Saver(max_to_keep=1)
        wall_t, total_step = self.load_checkpoint(sess, saver, checkpoint_dir)
        return wall_t, total_step, saver, checkpoint_dir

    def save_checkpoint(self, sess, saver, checkpoint_dir, wall_t, total_step, start_time):
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

    def push_stats_updates(self, stats, loss_arr, learning_rate, q_max_arr, epsilon_arr, action_arr, reward_arr, l_step, g_step):
        stats.update({'loss': np.average(loss_arr), 
                    'learning_rate': learning_rate,
                    'qmax': np.average(q_max_arr),
                    'epsilon': np.average(epsilon_arr),
                    'episode_actions': action_arr,
                    'reward': np.sum(reward_arr),
                    'steps': l_step,
                    'step': g_step
                    }) 

    def init_networks(self, sess, device, random_seed, action_size, gradient_clip_norm):
        o_network = DeepQNetwork(sess, device, 'online_network', random_seed, action_size, 
            trainable=True,
            gradient_clip_norm=gradient_clip_norm)
        t_network = DeepQNetwork(sess, device, 'target_network', random_seed, action_size)

        return o_network, t_network

    def train_network(self, states, actions, rewards, new_states, terminals, trainer_id):
        state_batch = np.empty((self.batch_size, 84, 84, 4), dtype=np.float16)
        action_batch = np.empty((self.batch_size, 3), dtype=np.float16)
        target_batch = np.empty((self.batch_size, 1))
        for n in range(len(states)):
            #s_t = states[n]
            #a_t = actions[n]
            r_t = rewards[n]
            s_t1 = new_states[n]
            terminal = terminals[n]

            if not terminal:
                q_t1 = self.agent.target_predict(s_t1)
                target = r_t + (self.gamma * np.max(q_t1))
            else:
                target = r_t

            #state_batch[n] = s_t
            #action_batch[n] = a_t
            target_batch[n] = target

        self.online_network.train(states, actions, target_batch, trainer_id)
        print('train done')
        #self.stats.training_count += 1
        #self.stats.frame_counter += 1
        self.stats.training_count.value += 1

main(args)


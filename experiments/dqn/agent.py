from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as numpy
import time

from settings import Settings
from game_state import GameState
from experience import Experience

class Agent(Process):
    def __init__(self, initial_epsilon, final_epsilon):

        self.final_epsilon = final_epsilon
        self.game_state = GameState(settings.random_seed, settings.game, settings.display)
        self.gamma = settings.gamma

    def anneal_epsilon(self, step):
        if step < Settings.epsilon_anneal:
            epsilon = 1.0 - step * ((1.0 - final_epsilon) / Settings.epsilon_anneal)
        else:
            epsilon = self.final_epsilon

    def predict(self, state):
        self.prediction_queue.put((self.index, state))
        q_values = self.wait_queue.get()
        return q_values

    def predict_target(self, state):
        self.target_prediction_queue.put((self.index, state))
        target_q_values = self.target_wait_queue.get()
        return target_q_values

    @staticmethod
    def select_action(epsilon, q_values):
        if Settings.display or np.random.random() < epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, ACTION_SIZE)
    @staticmethod
    def stack_data(s, a, r):
        return np.vstack(s), np.vstack(a), np.vstack(r)

    def play_episode(self):
        state = self.env.reset()
        terminal = final_epsilon
        state_batch, action_batch, target_batch = [], [], []
        epsilon = self.anneal_epsilon(STEP)

        local_step = 0

        action_arr, q_max_arr, reward_arr, epsilon_arr = [], [], [], [], []

        while not terminal:
            # Get the Q-values of the current state (s_t)
            q_values = self.predict(state)
            # Select action (a_t)
            action = self.select_action(epsilon, [state])
            # Make action (a_t) an observe (s_t1)
            new_state, reward, terminal = self.env.step(action)
            # Get the new state's Q-values
            q_values_new = self.predict_target(sess, [new_state])


            if settings.method.lower() == 'sarsa':
                # Get Q(s',a') for selected action to update Q(s,a)
                q_value_new = q_values_new[action]
            else:
                # Get max(Q(s',a')) to update Q(s,a)
                q_value_new = np.max(q_values_new)

            if not terminal: 
                # Q-learning: update with reward + gamma * max(Q(s',a')
                # SARSA: update with reward + gamma * Q(s',a') for the action taken in s' - not yet fully  tested
                value = reward + (settings.gamma * q_value_new)
            else:
                # Terminal state, update using reward
                value = reward

            onehot_action = np.eye(1, ACTION_SIZE, action)
            state_batch.append([state])
            action_batch.append(onehot_action)
            target_batch.append([update])

            # Save for stats
            action_arr.append(action)
            reward_arr.append(reward)
            q_max_arr.append(np.max(q_values))
            epsilon_arr.append(epsilon)

            if terminal or local_step % Settings.tmax == 0:
                #print 'pushing stats'
                #print 'Thread: {}  /  Global step: {}  /  Local steps: {}  /  Reward: {}  /  Qmax: {}  /  Epsilon: {}'.format(str(thread_index).zfill(2), 
                    #g_step, local_step, np.sum(reward_arr), format(np.average(q_max_arr), '.1f'), format(np.average(epsilon_arr), '.2f'))
                stacked_s, stacked_a, stacked_t = stack_data(state_batch, action_batch, target_batch)
                state_batch, action_batch, target_batch = [], [], []
                accumulated_r = np.sum(reward_arr)
                reward_arr = []
                yield stacked_s, stacked_a, stacked_t, accumulated_r

            else:
                state = new_state
                local_step += 1

        def run(self):
            time.sleep(1)
            np.random.seed(self.index + Settings.random_seed)

            while self.stop_flag.value == 0:
                total_length = 0
                total_reward = 0

            for s, a, r, accumulated_r in self.play_episode():
                total_reward += accumulated_r
                total_length += len(s)
                self.training_queue.put((s,a,t))
            self.episode_log_queue.put((datetime.now(), total_reward, total_length))
    

    
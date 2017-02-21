from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from game_state import GameState
from experience_replay import ExperienceReplayMemory

from stats import Stats

class Agent(Process):
    def __init__(self, prediction_queue, target_prediction_queue, training_queue, log_queue, experience_replay, epsilon_settings,
        random_seed=0, 
        game='BreakoutDeterministic-v0', 
        display=False, 
        no_op_max=30, 
        play_mode=False, 
        batch_size=32, 
        total_steps=0):
        super(Agent, self).__init__()

        np.random.seed(random_seed)

        self.prediction_queue = prediction_queue
        self.target_prediction_queue = target_prediction_queue
        self.training_queue = training_queue
        self.log_queue = log_queue
        self.play_mode = play_mode
        self.epsilon_settings = epsilon_settings
        self.batch_size = batch_size
        self.total_steps = total_steps

        self.game_state = GameState(random_seed, game, display, no_op_max)

        self.wait_queue = Queue(maxsize=1)
        self.target_wait_queue = Queue(maxsize=1)
        self.stop_flag = Value('i', 0)

    def anneal_epsilon(self, initial_epsilon, final_epsilon, current_step, anneal_steps):
        return initial_epsilon - current_step * ((initial_epsilon - final_epsilon) / float(anneal_steps))

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_queue.put(state)
        # wait for the prediction to come back
        q_values = self.wait_queue.get()
        return q_values

    def target_predict(self, state):
        # put the state in the prediction q
        self.target_prediction_queue.put(state)
        # wait for the prediction to come back
        q_values = self.target_wait_queue.get()
        return q_values

    def select_action(self, q_values, epsilon):
        if np.random.random() < epsilon or self.play_mode:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, self.game_state.action_size)
        return action, np.max(q_values)

    def run_episode(self, epsilon):
        state, reward, terminal = self.game_state.reset()

        total_reward = 0
        steps = 0
        qmax_array = []

        while not terminal:
            q_values = self.predict(state)
            action, qmax = self.select_action(q_values, epsilon)
            qmax_array.append(qmax)
            new_state, reward, terminal = self.env.step(action)
            
            total_reward += reward

            experience_replay.save(state, action, reward, new_state, terminal)
            self.training_queue.put(experience_replay.sample(self.batch_size))

            step += 1

            if terminal:
                return steps, total_reward, np.average(qmax_array)

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        while self.stop_flag.value == 0:
            epsilon = self.anneal_epsilon(self.epsilon_settings['initial_epsilon'], self.epsilon_settings['final_epsilon'], self.total_steps, self.epsilon_settings['anneal_steps'])
            steps, total_reward, qmax_average = self.run_episode(epsilon)
            self.total_steps += steps

            self.log_queue.put((datetime.now(), total_reward, steps))

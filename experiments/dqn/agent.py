from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from game_state import GameState
from experience_replay import ExperienceReplayMemory

from stats import Stats

class Agent(Process):
    def __init__(self, 
        prediction_queue, 
        target_prediction_queue, 
        training_queue, 
        log_queue, 
        experience_replay, 
        epsilon_settings,
        random_seed=0, 
        batch_size=32, 
        total_steps=0,
        play_mode=False,
        game='BreakoutDeterministic-v0', 
        display=False, 
        no_op_max=30):
        super(Agent, self).__init__()

        np.random.seed(random_seed)

        self.prediction_queue = prediction_queue
        self.target_prediction_queue = target_prediction_queue
        self.training_queue = training_queue
        self.log_queue = log_queue
        self.experience_replay = experience_replay
        self.epsilon_settings = epsilon_settings
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.play_mode = play_mode

        self.game_state = GameState(random_seed, game, display, no_op_max)

        self.wait_queue = Queue(maxsize=1)
        self.target_wait_queue = Queue(maxsize=1)
        self.stop_flag = Value('i', 0)
        print('agent stated')

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
        #qmax_array = []

        while not terminal:
            q_values = self.predict(state)
            action, qmax = self.select_action(q_values, epsilon)
            #qmax_array.append(qmax)
            new_state, reward, terminal = self.game_state.step(action)
            
            total_reward += reward

            self.experience_replay.save(state, action, reward, new_state, terminal)
            yield self.experience_replay.sample(self.batch_size)

            steps += 1
            print('s: {}, term: {}, rew: {}'.format(steps, terminal, reward))
            if terminal:
                #print('term')
                print('Steps {}, total_rew: {}'.format(steps, total_reward))
            #else:
            state = new_state

    def run_fill_episode(self):
        state, reward, terminal = self.game_state.reset()

        while not terminal:
            q_values = self.predict(state)
            action = np.random.randint(0, self.game_state.action_size)
            new_state, reward, terminal = self.game_state.step(action)
            
            self.experience_replay.save(state, action, reward, new_state, terminal)

            state = new_state

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(2)
        for n in range(5):
            print('Episode {}'.format(n))
            self.run_fill_episode()

        print('CURRENT ER SIZE: {}'.format(self.experience_replay.current_size))
        total_steps = 0
        while self.stop_flag.value == 0:
            epsilon = self.anneal_epsilon(
                self.epsilon_settings['initial_epsilon'], 
                self.epsilon_settings['final_epsilon'], 
                self.total_steps, 
                self.epsilon_settings['anneal_steps'])

            total_reward = 0
            for s_t, a_t, r_t, s_t1, term in self.run_episode(epsilon):
                total_reward += r_t
                total_steps += 1
                if term is None:
                    break
                self.training_queue.put((s_t, a_t, r_t, s_t1, term))
            #self.total_steps += steps
            print('total_steps: {}'.format(total_steps))
            print(' ')
            self.log_queue.put((datetime.now(), total_reward, total_steps))

import random
import numpy as np

class ExperienceReplayMemory(object):
    def __init__(self, experience_replay_size):
        self.experience_replay_size = experience_replay_size
        self.full_message = False
        self.reset()

    def reset(self):
        self.s_t_memory = np.empty((self.experience_replay_size, 84, 84, 4), dtype=np.float16)
        self.a_t_memory = np.empty((self.experience_replay_size, 3), dtype=np.float16)
        self.r_t_memory = np.empty(self.experience_replay_size)
        self.s_t1_memory = np.empty((self.experience_replay_size, 84, 84, 4), dtype=np.float16)
        self.terminal_memory = np.empty(self.experience_replay_size)
        self.current_size = 0

    def save(self, s_t, a_t, r_t, s_t1, terminal):
        self.s_t_memory[self.current_size] = s_t
        self.a_t_memory[self.current_size] = a_t
        self.r_t_memory[self.current_size] = r_t
        self.s_t1_memory[self.current_size] = s_t1
        self.terminal_memory[self.current_size] = terminal

        self.current_size = (self.current_size + 1) % self.experience_replay_size

    def sample(self, batch_size):
        indices = [np.random.randint(0, self.current_size) for n in range(min(batch_size, self.current_size))]
        return self.s_t_memory[indices], self.a_t_memory[indices], self.r_t_memory[indices], self.s_t1_memory[indices], self.terminal_memory[indices]

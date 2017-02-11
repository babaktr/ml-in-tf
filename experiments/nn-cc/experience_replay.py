import random

class ExperienceReplayMemory(object):
    def __init__(self, experience_replay_size):
        self.experience_replay_size = experience_replay_size

        self.s_t_memory = []
        self.a_t_memory = []
        self.r_t_memory = []
        self.s_t1_memory = []
        self.terminal_memory = []

    def save(self, s_t, a_t, r_t, s_t1, terminal):
        self.s_t_memory.insert(0, s_t)
        self.a_t_memory.insert(0, a_t)
        self.r_t_memory.insert(0, r_t)
        self.s_t1_memory.insert(0, s_t1)
        self.terminal_memory.insert(0, terminal)

        if len(self.s_t_memory) > self.experience_replay_size:
            self.s_t_memory.pop()
            self.a_t_memory.pop()
            self.r_t_memory.pop()
            self.s_t1_memory.pop()
            self.terminal_memory.pop()


    def sample(self, batch_size):
        indices = random.sample(range(len(self.s_t_memory)), min(batch_size, len(self.s_t_memory)))

        sample_s_t = []
        sample_a_t = []
        sample_r_t = []
        sample_s_t1 = []
        sample_terminal = []

        for index in indices:
            sample_s_t.append(self.s_t_memory[index])
            sample_a_t.append(self.a_t_memory[index])
            sample_r_t.append(self.r_t_memory[index])
            sample_s_t1.append(self.s_t1_memory[index])
            sample_terminal.append(self.terminal_memory[index])

        return sample_s_t, sample_a_t, sample_r_t, sample_s_t1, sample_terminal

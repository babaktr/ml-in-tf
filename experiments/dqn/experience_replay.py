import random

class ExperienceReplayMemory(object):
    def __init__(self, experience_replay_size):
        self.experience_replay_size = experience_replay_size
        self.full_message = False
        self.reset()

    def save(self, s_t, a_t, r_t, s_t1, terminal):
        self.s_t_memory.append(s_t)
        self.a_t_memory.append(a_t)
        self.r_t_memory.append(r_t)
        self.s_t1_memory.append(s_t1)
        self.terminal_memory.append(terminal)

        if len(self.s_t_memory) > self.experience_replay_size:
            if not self.full_message:
                print('Memory full! Started erasing old experiences.')
                self.full_message = True
            self.s_t_memory.pop(0)
            self.a_t_memory.pop(0)
            self.r_t_memory.pop(0)
            self.s_t1_memory.pop(0)
            self.terminal_memory.pop(0)

    def reset(self):
        self.s_t_memory = []
        self.a_t_memory = []
        self.r_t_memory = []
        self.s_t1_memory = []
        self.terminal_memory = []

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

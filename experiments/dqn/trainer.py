from threading import Thread
import numpy as np

from Config import Config


class Trainer(Thread):
    def __init__(self, server, batch_size):
        super(Trainer, self).__init__()
        self.setDaemon(True)

        self.server = server
        self.batch_size = batch_size
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            s_t, a_t, r_t, s_t1, terminal = self.server.training_queue.get()
            self.server.train_network(s_t, a_t, r_t, s_t1, terminal)

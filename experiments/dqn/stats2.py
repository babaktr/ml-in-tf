import sys
if sys.version_info >= (3,0):
    from queue import Queue as queueQueue
else:
    from Queue import Queue as queueQueue

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time


class Stats(Process):
    def __init__(self, total_steps):
        super(Stats, self).__init__()
        self.episode_log_queue = Queue(maxsize=100)
        self.total_steps = Value('i', 0)
        self.training_count = Value('i', 0)
        self.episode_count = Value('i', 0)
        self.stop_flag = Value('i', 0)
        self.total_frame_count = 0

    def FPS(self):
        return np.ceil(self.total_frame_count / (time.time() - self.start_time))

    def TPS(self):
        return np.ceil(self.training_count.value / (time.time() - self.start_time))

    def run(self):
        print('STATS START')
        with open('results.txt', 'a') as results_logger:
            self.start_time = time.time()
            first_time = datetime.now()
            while self.stop_flag.value == 0:
                print('RUN')
                episode_time, reward, steps = self.episode_log_queue.get()
                #results_logger.write('%s, %d, %d\n' % (episode_time.strftime("%Y-%m-%d %H:%M:%S"), reward, length))
                #results_logger.flush()

                self.total_steps.value += steps
                self.episode_count.value += 1

                print(
                    '[Time: %8d] '
                    '[Episode: %8d Score: %10.4f] '
                    '[PPS: %5d TPS: %5d] '
                    % (int(time.time()-self.start_time),
                       self.episode_count.value, reward,
                       self.FPS(), self.TPS()))
                sys.stdout.flush()


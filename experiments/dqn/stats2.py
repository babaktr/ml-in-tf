import sys
if sys.version_info >= (3,0):
    from queue import Queue as queueQueue
else:
    from Queue import Queue as queueQueue

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from Config import Config


class Stats(Process):
    def __init__(self, total_steps):
        self.episode_log_q = Queue(maxsize=100)
        self.total_steps = Value('i', 0)
        self.training_count = Value('i', 0)
        self.total_frame_count = 0

    def FPS(self):
        return np.ceil(self.total_frame_count / (time.time() - self.start_time))

    def TPS(self):
        return np.ceil(self.training_count.value / (time.time() - self.start_time))
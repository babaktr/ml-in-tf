from threading import Thread
import numpy as np

class PredictorTarget(Thread):
    def __init__(self, server, batch_size):
        super(PredictorTarget, self).__init__()
        self.setDaemon(True)
        self.batch_size = batch_size

        self.server = server
        self.stop_flag = False

    def run(self):
        states = np.zeros((self.batch_size, 84, 84, 4), dtype=np.float32)

        while not self.stop_flag:
            states[0] = self.server.target_prediction_queue.get()

            size = 1
            while size < self.batch_size and not self.server.target_prediction_queue.empty():
                states[size] = self.server.target_prediction_queue.get()
                size += 1

            batch = states[:size]
            p = self.server.target_network.predict(batch)

            for i in range(size):
                self.server.agent.target_wait_queue.put(p[i])

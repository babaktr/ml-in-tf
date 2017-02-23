from threading import Thread
import numpy as np

class PredictorOnline(Thread):
    def __init__(self, server, batch_size, id):
        super(PredictorOnline, self).__init__()
        self.setDaemon(True)
        self.batch_size = 2

        self.id = id
        self.server = server
        self.stop_flag = False

    def run(self):
        states = np.zeros((self.batch_size, 84, 84, 4), dtype=np.float32)

        while not self.stop_flag:
            states[0] = self.server.prediction_queue.get()

            size = 1
            while size < self.batch_size and not self.server.prediction_queue.empty():
                states[size] = self.server.prediction_queue.get()
                size += 1

            batch = states[:size]
            p = self.server.online_network.predict(batch)

            for i in range(size):
                self.server.agent.wait_queue.put(p[i])

from threading import Thread
import numpy as np

class predictor(Thread):
    def __init__(self, server, batch_size):
        self.setDaemon(True)
        self.batch_size = batch_size

        self.server = server
        self.stop_flag = False

    def run(self):
        states = np.zeros((self.batch_size, 84, 84, 4), dtype=np.float32)

        while not self.stop_flag:
            states[0] = self.server.prediction_queue.get()

            size = 1
            while size < Config.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
                ids[size], states[size] = self.server.prediction_q.get()
                size += 1

            batch = states[:size]
            p, v = self.server.model.predict_p_and_v(batch)

            for i in range(size):
                if ids[i] < len(self.server.agents):
                    self.server.agents[ids[i]].wait_q.put((p[i], v[i]))

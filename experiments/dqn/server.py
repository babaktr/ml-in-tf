import settings
import tensorflow as tf
import time

from agent import Agent
from game_state import GameState
from multiprocessing import Queue
from networks import Networks
from stats import Stats
from ThreadDynamicadjustment import ThreadDynamicadjustment
from ThreadTrainer import ThreadTrainer


class ParameterServer:
	def __init__(self, settings)
		#self.stats = stats
		#
		self.training_queue = Queue(max_size=settings.max_queue_size)
		self.prediction_queue = Queue(max_size=settings.max_queue_size)

		self.networks = Networks(Settings.device, action_size)

		self.stats.episodes.value = self.networks.load_checkpoint()

		self.training_step = 0
		self.frame_step = 0

		self.agents = []
		self.predictors = []
		self.trainers = []
		self.dynamic_adjustment = threadDynamicadjustment(self)

	def train(self, s, a, r, trainer_index):
		self.networks.train(s, a, r, trainer_index)
		self.training_step += 1
		self.frame_step += 4

		self.stats.training_count.value += 1
		self.dynamic_adjustment.temporal_training_count += 1

		if settings.save_stats and self.stats.training_count.value % settings.stats_update_prequency == 0:
			self.network.log(s, a, r)

	def save_checkpoint(self):
		self.networks.save(self.stats.episodes.value)

	def increase_agents(self):
		self.agents.append(Agent(len(self.agents), self.prediction_queue, self.training_queue, self.stats_episode_log_queue))
		self.agents[-1].start()

	def increase_predictors(self):
		self.predictors.append(ThreadPredictor(self, len(self.predictors)))
		self.predictors[-1].start()

	def increase_trainers(self):
		self.trainers.append(ThreadTrainer(self, len(self.trainers)))
		self.trainers[-1].start()

	def decrease_predictors(self):
		self.predictors[-1].stop_flag = True
		self.predictors[-1].join()
		self.predictors.pop()

	def decrease_agents(self):
		self.agents[-1].stop_flag.value = True
		self.agents[-1].join()
		self.agents.pop()

	def decrease_trainers(self):
		self.trainers[-1].stop_flag = True
		self.trainers[-1].join()
		self.trainers.pop()

	def disable_trainers(self):
		for trainer in self.trainers:
			trainer.enabled = False

	def anneal_learning_rate(step):
        return settings.learning_rate - (step * (settings.learning_rate / settings.max_global_steps))


	def main(self):
		self.stats.start()
		self.dynamic_adjustment.start()

		if settings.display:
			self.disable_trainers()

		while self.global_step.value < settings.max_global_steps:
			self.networks.learning_rate =  self.anneal_learning_rate(self.global_step.value)

			if self.stats.saving_model.value > 0:
				self.save_checkpoint()
				self.satts.saving_model.value = 0

			time.sleep(0.01)

		self.dynamic_adjustment.stop_flag = True

		print('Removing agents.')
		while self.agents:
			self.decrease_agents()

		print('Removing predictors.')
		while self.predictors:
			self.decrease_predictors()

		print('Removing trainers.')
		while self.trainers:
			self.decrease_trainers()



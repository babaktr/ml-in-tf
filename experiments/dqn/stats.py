import tensorflow as tf
import numpy as np

class Stats(object):  
    def __init__(self, sess, summary_writer, histogram_summary):
        self.sess = sess

        self.histogram_summary = histogram_summary
        self.histogram_summary_count = 0

        self.writer = summary_writer

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['actions/0', 'actions/1', 'actions/2', 
                                'network/loss', 'network/learning_rate', 
                                'episode/avg_q_max', 'episode/epsilon', 'episode/reward', 'episode/steps',
                                'evaluation/rewards', 'evaluation/score', 'evaluation/steps']

            self.summary_placeholders = {}
            self.summary_ops = {}

        for tag in scalar_summary_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
        
        with tf.variable_scope('histogram'):
            histogram_summary_tags = ['episode/episode_rewards', 'episode/episode_steps', 'episode/episode_actions']

            self.histogram_placeholders = {}
            self.histogram_ops = {}

        for tag in histogram_summary_tags:
            self.histogram_placeholders[tag] = tf.placeholder('float32', None, name=tag)
            self.histogram_ops[tag] = tf.summary.histogram(tag, self.histogram_placeholders[tag])

        self.reset_saved_values()

    def reset_saved_values(self):
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_steps = []

    def update(self, dictionary):
        # Save for histogram update
        self.episode_rewards.append(dictionary['reward'])
        self.episode_steps.append(dictionary['steps'])
        actions = dictionary['episode_actions']
        self.episode_actions = self.episode_actions + actions

        self.inject_summary({'network/loss': dictionary['loss'],
                            'network/learning_rate': dictionary['learning_rate'],
                            'episode/avg_q_max': dictionary['qmax'],
                            'episode/epsilon': dictionary['epsilon'],
                            'episode/reward': dictionary['reward'],
                            'episode/steps':dictionary['steps'],
                            'actions/0': float(actions.count(0))/len(actions),
                            'actions/1': float(actions.count(1))/len(actions),
                            'actions/2': float(actions.count(2))/len(actions)
                            }, dictionary['step'])

        if self.histogram_summary_count % self.histogram_summary == 0:
            self.inject_histogram({'episode/episode_rewards': self.episode_rewards,
                                'episode/episode_steps': self.episode_steps,
                                'episode/episode_actions': self.episode_actions 
                                }, dictionary['step'])

            self.reset_saved_values()

        self.histogram_summary_count += 1

    def update_eval(self, dictionary):
        self.inject_summary({
              'evaluation/rewards': dictionary['rewards'],
              'evaluation/score': dictionary['score'],
              'evaluation/steps': dictionary['steps'],
            }, dictionary['step'])

    def inject_summary(self, tag_dict, t):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
          self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
          self.writer.add_summary(summary_str, t)
          self.writer.flush()

    def inject_histogram(self, tag_dict, t):
        histogram_str_lists = self.sess.run([self.histogram_ops[tag] for tag in tag_dict.keys()], {
          self.histogram_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for histogram_str in histogram_str_lists:
          self.writer.add_summary(histogram_str, t)
          self.writer.flush()


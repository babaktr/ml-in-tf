import tensorflow as tf
import numpy as np

class Stats(object):  
  def __init__(self, sess, summary_writer, stat_level):
    self.sess = sess
    self.stat_level = stat_level

    self.writer = summary_writer
    with tf.variable_scope('summary'):
        if stat_level == 1:
            scalar_summary_tags = ['loss']
        elif stat_level == 2:
            scalar_summary_tags = ['network/loss', 'network/accuracy']
        elif stat_level == 3:
            scalar_summary_tags = ['episode/avg_q_max', 'episode/epsilon', 'episode/reward', 'episode/steps']
        elif stat_level == 4:
            scalar_summary_tags = ['network/loss', 'network/accuracy', 'episode/avg_q_max', 'episode/epsilon', 'episode/reward', 'episode/steps']

        self.summary_placeholders = {}
        self.summary_ops = {}

    for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.scalar(tag, self.summary_placeholders[tag])

  def update(self, dictionary):
    if self.stat_level == 1:
      self.inject_summary({
          'loss': dictionary['loss']
      }, dictionary['step'])
    elif self.stat_level == 2:
      self.inject_summary({
          'network/loss': dictionary['loss'],
          'network/accuracy': dictionary['accuracy']
      }, dictionary['step'])
    elif self.stat_level == 3:
      self.inject_summary({
            'episode/avg_q_max': dictionary['qmax'],
            'episode/epsilon': dictionary['epsilon'],
            'episode/reward': dictionary['reward'],
            'episode/steps':dictionary['steps']
        }, dictionary['step'])
    else: # self.stat_level == 4
      self.inject_summary({
            'network/loss': dictionary['loss'],
            'network/accuracy': dictionary['accuracy'],
            'episode/avg_q_max': dictionary['qmax'],
            'episode/epsilon': dictionary['epsilon'],
            'episode/reward': dictionary['reward'],
            'episode/steps':dictionary['steps']
        }, dictionary['step'])

  def inject_summary(self, tag_dict, t):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, t)
      self.writer.flush()


import tensorflow as tf
import numpy as np

class Stats(object):  
  def __init__(self, sess, summary_writer, stat_level):
    self.sess = sess
    self.stat_level = stat_level

    self.writer = summary_writer
    with tf.variable_scope('summary'):
        if stat_level == 1:
            scalar_summary_tags = ['error']

        self.summary_placeholders = {}
        self.summary_ops = {}

    for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.scalar(tag, self.summary_placeholders[tag])


  def update(self, error, epoch):
    self.inject_summary({
        'error': error
    }, epoch)

  def inject_summary(self, tag_dict, t):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, t)
      self.writer.flush()


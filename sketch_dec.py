import tensorflow as tf
import numpy as np
import linreg_nn
import sys


ckpt_path = './linreg_log/test1/'

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

def load_ckpt(sess, checkpoint_path):
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
  saver.restore(sess, ckpt.model_checkpoint_path)

def decode(z_input=None, draw_mode=True, temperature=0.1, factor=0.2):
  z = None
  if z_input is not None:
    z = [z_input]
  sample_strokes, m = sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
  strokes = to_normal_strokes(sample_strokes)
  if draw_mode:
    draw_strokes(strokes, factor)
  return strokes

# for v in tf.get_default_graph().as_graph_def().node:
#   print v.name

load_ckpt(sess, ckpt_path)
# z = 
# _ = decode(z, temperature=0.1)
"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import cifar10_input

class LinfPGDAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess, is_training):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = x_nat.astype(np.float)

    for i in range(self.num_steps):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.is_training: is_training,
                                            self.model.y_input: y})

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x

  def _label_smoothing(self, one_hot, factor):
    return one_hot * factor + (one_hot - 1.) * ((factor-1) / float(self.model.num_classes - 1))

  def perturb_avmixup(self, x_nat, y, gamma, lambda1, lambda2, sess, is_training):
    """Given a set of examples (x_nat, y), returns a set of training
       examples constructed by AVmixup."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = x_nat.astype(np.float)

    for i in range(self.num_steps):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.is_training: is_training,
                                            self.model.y_input: y})

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    perturb = (x - x_nat) * gamma
    adversarial_vertex = x_nat + perturb
    adversarial_vertex = np.clip(adversarial_vertex, 0, 255) # ensure valid pixel range
    y_nat = self._label_smoothing(y, lambda1)
    y_vertex = self._label_smoothing(y, lambda2)
    x_weight = np.random.beta(1.0, 1.0, [x.shape[0], 1, 1, 1])
    y_weight = np.reshape(x_weight, [-1, 1])
    x = x_nat * x_weight + adversarial_vertex * (1-x_weight)
    y = y_nat * y_weight + y_vertex * (1-y_weight)
    return [x, y]


if __name__ == '__main__':
  import json
  import sys
  import math


  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model(num_classes=10)
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  data_path = config['data_path']
  cifar = cifar10_input.CIFAR10Data(data_path)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]
      y_batch = np.eye(10)[y_batch]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess, is_training=False)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))

"""A GRU layer."""
import numpy
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T

from rnnlayer import RNNLayer

class GRULayer(RNNLayer):
  """A GRU layer.

  Parameter names follow convention in Richard Socher's CS224D slides.
  """
  def create_vars(self, create_init_state):
    # Initial state
    if create_init_state:
      self.h0 = theano.shared(
          name='h0', 
          value=0.1 * numpy.random.uniform(-1.0, 1.0, self.nh).astype(theano.config.floatX))
      init_state_params = [self.h0]
    else:
      init_state_params = []

    # Encoder hidden state updates
    self.w_z = theano.shared(
        name='w_z',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.u_z = theano.shared(
        name='u_z',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.w_r = theano.shared(
        name='w_r',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.u_r = theano.shared(
        name='u_r',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.w = theano.shared(
        name='w',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.u = theano.shared(
        name='u',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    recurrence_params = [self.w_z, self.u_z, self.w_r, self.u_r, self.w, self.u]

    # Params
    self.params = init_state_params + recurrence_params

  def get_init_state(self):
    return self.h0

  def step(self, input_t, h_prev):
    ######  Fill Your Answer  ######
    z_t  = T.nnet.sigmoid(T.dot(input_t, self.w_z) + T.dot(h_prev, self.u_z))
    r_t  = T.nnet.sigmoid(T.dot(input_t, self.w_r) + T.dot(h_prev, self.u_r))
    nh_t = T.tanh(r_t * T.dot(h_prev, self.u) + T.dot(input_t, self.w))
    h_t  = (1 - z_t) * nh_t + z_t * h_prev

    return h_t

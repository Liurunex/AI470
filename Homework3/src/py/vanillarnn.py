"""A vanilla RNN layer."""
import numpy
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T

from rnnlayer import RNNLayer

class VanillaRNNLayer(RNNLayer):
  """A standard vanilla RNN layer."""
  def create_vars(self, create_init_state):
    # Initial state
    if create_init_state:
      self.h0 = theano.shared(
          name='h0', 
          value=0.1 * numpy.random.uniform(-1.0, 1.0, self.nh).astype(theano.config.floatX))
      init_state_params = [self.h0]
    else:
      init_state_params = []

    # Recurrent layer
    self.w_hx = theano.shared(
        name='w_hx',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.w_hh = theano.shared(
        name='w_hh',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    recurrence_params = [self.w_hx, self.w_hh]

    # Params
    self.params = init_state_params + recurrence_params

  def get_init_state(self):
    return self.h0

  def step(self, input_t, h_prev):
    h_t = T.nnet.sigmoid(T.dot(h_prev, self.w_hh) + T.dot(input_t, self.w_hx))
    return h_t

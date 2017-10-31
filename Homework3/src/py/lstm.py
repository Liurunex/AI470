"""An LSTM layer."""
import numpy
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T

from rnnlayer import RNNLayer

class LSTMLayer(RNNLayer):
  """An LSTM layer.

  Parameter names follow convention in Richard Socher's CS224D slides.
  """
  def create_vars(self, create_init_state):
    # Initial state
    # The hidden state must store both c_t, the memory cell, 
    # and h_t, what we normally call the hidden state
    if create_init_state:
      self.h0 = theano.shared(
          name='h0', 
          value=0.1 * numpy.random.uniform(-1.0, 1.0, 2 * self.nh).astype(theano.config.floatX))
      init_state_params = [self.h0]
    else:
      init_state_params = []

    # Recurrent layer 
    self.w_i = theano.shared(
        name='w_i',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.u_i = theano.shared(
        name='u_i',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.w_f = theano.shared(
        name='w_f',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.u_f = theano.shared(
        name='u_f',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.w_o = theano.shared(
        name='w_o',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.u_o = theano.shared(
        name='u_o',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.w_c = theano.shared(
        name='w_c',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.u_c = theano.shared(
        name='u_c',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    recurrence_params = [
        self.w_i, self.u_i, self.w_f, self.u_f,
        self.w_o, self.u_o, self.w_c, self.u_c,
    ]

    # Params
    self.params = init_state_params + recurrence_params

  def unpack(self, hidden_state):
    c_t = hidden_state[0:self.nh]
    h_t = hidden_state[self.nh:]
    return (c_t, h_t)

  def pack(self, c_t, h_t):
    return T.concatenate([c_t, h_t])

  def get_init_state(self):
    return self.h0

  def step(self, input_t, c_h_prev):
    c_prev, h_prev = self.unpack(c_h_prev)
    ######  Fill Your Answer  ######
    i_t = T.nnet.sigmoid(T.dot(input_t, self.w_i) + T.dot(h_prev, self.u_i))
    f_t = T.nnet.sigmoid(T.dot(input_t, self.w_f) + T.dot(h_prev, self.u_f))
    o_t = T.nnet.sigmoid(T.dot(input_t, self.w_o) + T.dot(h_prev, self.u_o))
    nc_t = T.tanh(T.dot(input_t, self.w_c) + T.dot(h_prev, self.u_c))
    c_t = f_t * c_prev + i_t * nc_t
    h_t = o_t * T.tanh(c_t)

    return self.pack(c_t, h_t)

  def get_h_for_write(self, c_h_t):
    c_t, h_t = self.unpack(c_h_t)
    return h_t


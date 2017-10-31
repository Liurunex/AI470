"""An output layer."""
import numpy
import theano
from theano.ifelse import ifelse
import theano.tensor as T

class OutputLayer(object):
  """Class that sepcifies parameters of an output layer.
  
  Conventions used by this class (shared with spec.py):
    nh: dimension of hidden layer
    nw: number of words in the vocabulary
    de: dimension of word embeddings
  """ 
  def __init__(self, vocab, hidden_size):
    self.vocab = vocab
    self.de = vocab.emb_size
    self.nh = hidden_size
    self.nw = vocab.size()
    self.create_vars()

  def create_vars(self):
    self.w_out = theano.shared(
        name='w_out', 
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nw, self.nh)).astype(theano.config.floatX))
        # Each row is one word
    self.params = [self.w_out]

  def write(self, h_t, attn_scores=None):
    """Get a distribution over words to write.
    
    Entries in [0, nw) are probablity of emitting i-th output word,
    and entries in [nw, nw + len(attn_scores))
    are probability of copying the (i - nw)-th word.

    Args:
      h_t: theano vector representing hidden state
      attn_scores: unnormalized scores from the attention module, if doing attention-based copying.
    """
    if attn_scores:
      ######  Fill Your Answer  ######

      # 1. calcuate the probablity distrbution over the vocabulary
      # 2. concatenate the result from 1. and attn_scores (in this order)
      # 3. normalize the concatenated scores by softmax
      pb  = T.dot(h_t, self.w_out.T)
      tem = T.concatenate([pb, attn_scores], axis=0)
      return T.nnet.softmax(tem)[0]

      return scores_normalized
    else:
      return T.nnet.softmax(T.dot(h_t, self.w_out.T))[0]

import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train   = X.shape[0]
  
  for i in range(num_train):
    scores = X[i].dot(W)
    
    scores -= max(scores) # shift max(Fi) for stablility on division of exp
    loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))

    for j in range(num_classes):
      softmax_res = np.exp(scores[j]) / np.sum(np.exp(scores))
      if j == y[i]:
        dW[:,y[i]] -= X[i]  
        dW[:,y[i]] += softmax_res * X[i]
      else:
        dW[:,j] += softmax_res * X[i]
    
  loss /= num_train
  dW   /= num_train

  loss += reg * np.sum(W * W)
  dW   += reg * W
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  scores = np.dot(X, W)
  max_scores = np.max(scores, axis = 1).reshape(num_train,1)
  scores -= max_scores

  loss = -np.sum(scores[np.arange(num_train),y]) + np.sum(np.log(np.sum(np.exp(scores), axis = 1)))
  
  softmax_res = np.exp(scores) / (np.sum(np.exp(scores), axis = 1)+1e-8).reshape(num_train,1)
  softmax_res[np.arange(num_train),y] -= 1
  dW = np.dot(X.T, softmax_res)

  loss /= num_train
  dW   /= num_train

  loss += reg * np.sum(W * W)
  dW   += reg * W
  
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


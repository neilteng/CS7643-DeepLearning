import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  dim, num_train = X.shape
  num_classes, _ = W.shape
  assert(dim == _)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  logits = W.dot(X)

  # Use the fact that softmax(z) = softmax(z+c)
  logits= logits - np.max(logits, axis=0)

  softmax_sum = np.sum(np.exp(logits), axis=0, keepdims=True)

  loss = - 1.0 / num_train * np.sum(logits[y, np.arange(num_train)] - np.log(softmax_sum)) + 0.5 * reg * np.sum(np.multiply(W,W))

  exp_scores =  np.exp(logits)
  probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
  probs[y, np.arange(num_train)] -= 1

  dW = reg * W  + 1.0 / num_train * probs.dot(X.T)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

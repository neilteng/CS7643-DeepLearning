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

  softmax_sum = np.sum(np.exp(logits), axis=0)

  loss = - 1.0 / num_train * np.sum(np.choose(y, logits) - np.log(softmax_sum)) + reg * np.sum(np.multiply(W,W))

  zeros_padding = np.zeros(shape=(num_classes, num_train))
  intermediate =  np.exp(logits)
  intermediate = 1 - intermediate / np.sum(intermediate, axis=0)
  zeros_padding[y, np.arange(num_train)] = np.choose(y, intermediate)

  dW = 2.0 * reg * W  - 1.0 / num_train * zeros_padding.dot(X.T)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

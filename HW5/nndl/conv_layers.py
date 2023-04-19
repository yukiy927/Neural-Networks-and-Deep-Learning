import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  #shape of data
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape

  #output size of conv layer
  H0 = int(1 + (H + 2 * pad - HH) / stride)
  W0 = int(1 + (W + 2 * pad - WW) / stride)

  #padded data
  pdm = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
  #initial output
  out = np.zeros((N, F, H0, W0))

  #conv layer
  for n in range(N):
    for f in range(F):
      for i in range(H0):
        for j in range(W0):
          out[n,f,i,j] = np.sum(pdm[n,:,i*stride:i*stride+HH,j*stride:j*stride+WW] * w[f]) + b[f]

  pass
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  #initialization
  dx = np.zeros(x.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)

  # pad gradient to x
  pdx = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

  for i, pddp in enumerate(xpad):
    for j, filter in enumerate(w):
      for xp in range(dout.shape[2]):
        xos = xp * stride
        for yp in range(dout.shape[3]):
          yos = yp * stride
          dw[j] += dout[i, j, xp, yp] * pddp[:, xos:xos + w.shape[2], yos: yos + w.shape[3]]
          pdx[i, :, xos:xos + w.shape[2], yos: yos + w.shape[3]] += dout[i, j, xp, yp] * w[j]

  db = np.sum(np.sum(np.sum(dout, axis=3), axis=2), axis=0)
  dx = pdx[:, :, pad:-pad, pad:-pad]

  pass

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  # parameters setup
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape

  #output size
  output_height = int((H - pool_height) / stride + 1)
  output_width = int((W - pool_width) / stride + 1)
  out = np.zeros((N,C,output_height,output_width))

  #max pooling
  for i in range(output_height):
    for j in range(output_width):
      out[:, :, i, j] = np.max(
        x[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width],
        axis=(2, 3)
      )
  pass
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  #initialization
  dx = np.zeros(x.shape)

  for i, dp in enumerate(x):
    for l, layer in enumerate(dp):
      for xp in range(dout.shape[2]):
        xos = xp * pool_param['stride']
        for yp in range(dout.shape[3]):
          yos = yp * pool_param['stride']
          field = layer[xos:xos + pool_param['pool_height'], yos:yos + pool_param['pool_width']]
          x_max, y_max = np.unravel_index(np.argmax(field, axis=None), field.shape)

          dx[i, l, x_max + xos, y_max + yos] = dout[i, l, xp, yp]

  pass

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  reshaped_array = np.reshape(x, (N * H * W, C))
  out, cache = batchnorm_forward(reshaped_array, gamma, beta, bn_param)
  out = np.reshape(out,(N, C, H, W))
  pass

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dout_rs = np.reshape(dout,(N * H * W, C))
  dx, dgamma, dbeta = batchnorm_backward(dout_rs, cache)
  dx = np.reshape(dx,(N, C, H, W))
  pass

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta
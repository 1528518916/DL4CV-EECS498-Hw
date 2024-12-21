import torch
import numpy as np
from convolutional_networks import BatchNorm

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_batchnorm_forward():
  # Create random input data
  N, D = 4, 5
  x = torch.randn(N, D)
  gamma = torch.ones(D)
  beta = torch.zeros(D)
  
  # Create parameters for batchnorm
  bn_param = {'mode': 'train'}
  
  # Compute forward pass
  out, cache = BatchNorm.forward(x, gamma, beta, bn_param)
  
  # Check that means are close to 0 and stds are close to 1
  out_mean = out.mean(dim=0)
  out_std = out.std(dim=0, unbiased=True)
  
  assert torch.allclose(out_mean, torch.zeros_like(out_mean), atol=1e-6), \
    f"Output means should be close to 0, got {out_mean}"
  assert torch.allclose(out_std, torch.ones_like(out_std), atol=1e-6), \
    f"Output stds should be close to 1, got {out_std}"

def test_batchnorm_backward():
  N, D = 4, 5
  x = torch.randn(N, D, requires_grad=True)
  gamma = torch.ones(D, requires_grad=True)
  beta = torch.zeros(D, requires_grad=True)
  
  # Forward pass
  bn_param = {'mode': 'train'}
  out, cache = BatchNorm.forward(x, gamma, beta, bn_param)
  
  # Compute gradients using PyTorch's autograd
  dout = torch.randn_like(out)
  out.backward(dout)
  
  # Compute gradients using our implementation
  dx, dgamma, dbeta = BatchNorm.backward(dout, cache)
  
  # Compare gradients
  dx_auto = x.grad
  dgamma_auto = gamma.grad
  dbeta_auto = beta.grad
  
  assert torch.allclose(dx, dx_auto, atol=1e-6), \
    f"dx error. Max diff: {torch.max(torch.abs(dx - dx_auto))}"
  assert torch.allclose(dgamma, dgamma_auto, atol=1e-3), \
    f"dgamma error. Max diff: {torch.max(torch.abs(dgamma - dgamma_auto))}"
  assert torch.allclose(dbeta, dbeta_auto, atol=1e-3), \
    f"dbeta error. Max diff: {torch.max(torch.abs(dbeta - dbeta_auto))}"

def test_batchnorm_inference():
  N, D = 4, 5
  x = torch.randn(N, D)
  gamma = torch.ones(D)
  beta = torch.zeros(D)
  
  # Train mode first to accumulate running statistics
  bn_param = {'mode': 'train', 'running_mean': torch.zeros(D), 'running_var': torch.ones(D)}
  out_train, _ = BatchNorm.forward(x, gamma, beta, bn_param)
  
  # Test mode
  bn_param['mode'] = 'test'
  out_test, _ = BatchNorm.forward(x, gamma, beta, bn_param)
  
  assert not torch.allclose(out_train, out_test), \
    "Training and test outputs should differ due to using running statistics"

if __name__ == '__main__':
  test_batchnorm_forward()
  test_batchnorm_backward()
  test_batchnorm_inference()
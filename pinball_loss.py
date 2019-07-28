import torch
import numpy as np

class PinballLoss():
  def __init__(self, quantile=0.10, reduction='none'):
      self.quantile = quantile
      assert 0 < self.quantile
      assert self.quantile < 1
      self.reduction = reduction
  def __call__(self, output, target):
      assert output.shape == target.shape
      loss = torch.zeros_like(target, dtype=torch.float)
      error = output - target
      smaller_index = error < 0
      bigger_index = 0 < error
      loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
      loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])

      if self.reduction == 'sum':
        loss = loss.sum()
      if self.reduction == 'mean':
        loss = loss.mean()

      return loss
'''
y = torch.tensor(np.arange(1000), dtype=torch.float)/10.
t = torch.tensor(np.ones(1000), dtype=torch.float) * 50

criterion = PinballLoss(quantile=0.05)
error, loss = criterion(y, t)
import matplotlib.pyplot as plt
plt.scatter(error, loss)
'''

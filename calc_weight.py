import torch

def calc_weight(Tb, n_class=2):
  weights = torch.zeros_like(Tb).to(torch.float)
  counts = torch.zeros(n_class)
  for i in range(n_class):
    counts[i] += (Tb == i).sum()
  probs = counts / counts.sum()
  weights = probs ** (-1)
  return weights

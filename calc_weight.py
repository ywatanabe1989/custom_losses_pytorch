import torch

def calc_weight(Tb, n_class):
  weights = torch.zeros_like(Tb).to(torch.float)
  recip_probs = torch.zeros_like(Tb).to(torch.float)
  counts = torch.zeros(n_class)

  for i in range(n_class):
    counts[i] += (Tb == i).sum()
  probs = counts / counts.sum()
  recip_probs = probs ** (-1)

  for i in range(n_class):
    mask = (Tb == i)
    weights[mask] = recip_probs[i]

  weights /= weights.mean() # normalize for stabilizing loss across batches
  return weights

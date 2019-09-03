import math
import torch

def pdf(x, mu, sigma): # probability density function
  var = sigma**2
  return 1./((2*math.pi*var)**0.5) * torch.exp(-(x-mu)**2 / (2*var))

def PDFLoss(x, mu, sigma, epsilon=1e-5):
  return - torch.log(pdf(x, mu, sigma) + epsilon)

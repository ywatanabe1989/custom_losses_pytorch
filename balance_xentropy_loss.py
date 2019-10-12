import torch
import torch.nn as nn

class BalanceCrossEntropyLoss():
  """Balance the sampling number disparity of the cross entropy loss

  -- Example
  n_classes = 4
  balance_loss = BalanceCrossEntropyLoss(n_classes)
  xentropy_criterion = nn.CrossEntropyLoss(reduction='none') # NOTE

  for _ in range(100):
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(n_classes)
    xentropy_loss = xentropy_criterion(input, target)
    balanced_loss = balance_loss(xentropy_loss, target) # here, just after calculating the cross entropy loss
    # print(balance_loss.n_samples_per_class)
  """
  def __init__(self, n_classes_int):
    self.n_classes_int = n_classes_int
    self.cum_n_samp_per_cls = torch.zeros(n_classes_int)

  def __call__(self, loss, Tb):
    # update counts arr
    for i in range(len(self.cum_n_samp_per_cls)):
      self.cum_n_samp_per_cls[i] += (Tb == i).sum()

    weights = torch.zeros_like(Tb, dtype=torch.float)
    probs_arr = 1. * self.cum_n_samp_per_cls / self.cum_n_samp_per_cls.sum()
    non_zero_mask = (probs_arr > 0)
    recip_probs_arr = torch.zeros_like(probs_arr)
    recip_probs_arr[non_zero_mask] = probs_arr[non_zero_mask] ** (-1)
    for i in range(self.n_classes_int):
      mask = (Tb == i)
      weights[mask] += recip_probs_arr[i]
    weights_norm = (weights / weights.mean()).to(loss.dtype).to(loss.device)
    loss *= weights_norm
    return loss


if __name__ == '__main__':
  n_classes = 3
  balance_loss = BalanceCrossEntropyLoss(n_classes)
  loss_orig = torch.Tensor([1,1,1,1,1,1])
  loss = loss_orig.clone()
  targets = torch.LongTensor([0,0,0,1,1,2])
  balanced_loss = balance_loss(loss, targets) # , cum_n_samp_per_cls=cum_n_samp_per_cls, n_classes_int=n_classes_int)
  print('Original Loss {}'.format(loss_orig))
  print('Balanced Loss {}'.format(balanced_loss))
  print()
  print('Original Loss mean {}'.format(loss_orig.mean()))
  print('Balanced Loss mean {}'.format(balanced_loss.mean()))
  print()
  print('Cumulated n_classes for balancing {}'.format(balance_loss.cum_n_samp_per_cls))

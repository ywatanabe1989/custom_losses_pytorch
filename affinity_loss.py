import torch
import torch.nn as nn

class AffinityLossLayer(nn.Module):
    '''
    https://arxiv.org/pdf/1901.07711.pdf

    Hyperparameters

    sigma:
        Sigma determines the cluster spread and helps achieve uniform intra-class variations,
        with larger sigma indicating larger variance for each cluster.
        [5, 20] showed optimal performance in the paper on an imbalanced-modified MNIST.

    lambda_:
        The enforced margine.

    t:
        t is between [1, m]. m projection vectors per class are learned
        in the novel multi-centered learning paradigm
        based on the proposed max-margin framework.
        # Not Implemented Yet.
    '''
    def __init__(self, n_in, n_out, lambda_=1., sigma=10., t=1, reduction='none'):
      super(AffinityLossLayer, self).__init__()
      self.params = nn.ParameterDict({
          'W': nn.Parameter(torch.randn(n_in, n_out), requires_grad=True) # all class centers
          })

      self.C = n_out
      self.lambda_ = lambda_
      self.sigma = sigma
      self.reduction = reduction

    def forward(self, y_pred, y_true):
        y_true_mask = self._to_onehot(y_true, self.C).bool().cuda()

        ## Max Margin Loss Function
        distance_all = torch.exp(- (y_pred.unsqueeze(-1) - self.params['W']).norm(dim=1) \
                     / self.sigma )
        distance_true = distance_all * y_true_mask
        distance_others = distance_all * ~y_true_mask

        Lmm = torch.max(torch.tensor(0.).cuda(), self.lambda_ + distance_others - distance_true)
        Lmm = Lmm.sum(axis=1)
        # fixme, multi-centered learning paradigm, Eq. (9) is not implemented yet.

        ## Rw
        Ws = torch.stack([self.params['W'].roll(i, dims=1) - self.params['W']
                          for i in range(self.C)]) # [self.C, n_in, self.C]

        mask = self.triu_mask(Ws)
        Ws *= mask.unsqueeze(1)
        N = mask.sum()
        w_norms = Ws.norm(dim=1)
        mu = ( 2 / (self.C**2 - self.C) ) * w_norms.sum()
        Rw = (((w_norms - mu)**2 )*mask).sum() / N

        loss = Lmm + Rw

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
        
    def _to_onehot(self, label, k):
        return torch.eye(k)[label].int()
        
    def _triu_mask(self, value):
        import torch
        n = value.size(-1)
        coords = value.new(n)
        torch.arange(n, out=coords)
        return  coords < coords.view(n, 1)

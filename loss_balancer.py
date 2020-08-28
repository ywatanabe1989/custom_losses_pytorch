import torch
import torch.nn as nn


class LossBalancer():
    """Balance the cost originated from imbalanced dataset in a online manner.

    # Example
    n_classes = 4
    balancer = LossBalancer(n_classes)
    xentropy_criterion = nn.CrossEntropyLoss(reduction='none')

    for _ in range(100):
        input = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, dtype=torch.long).random_(n_classes)
        loss = xentropy_criterion(input, target)
        balanced_loss = balancer(loss, target, train=True)
        balanced_loss = balanced_loss.mean() # Loss should be scaler to backprop
        # balanced_loss.backward() # when you train the model
    """
    def __init__(self, n_classes_int):
        self.n_classes_int = n_classes_int
        self.cum_n_samp_per_cls = torch.zeros(n_classes_int)
        self.weights_norm = torch.ones(n_classes_int) / n_classes_int

    def __call__(self, loss, Tb, train=True):
        if train is True:
            self._update_n_sample_counter(Tb)
        else:
            pass

        self._update_weights(loss, Tb)

        ## Balancing the Loss
        loss *= self.weights_norm
        return loss

    def _update_n_sample_counter(self, Tb):
        ## Update Counter of Sample Numbers on Each Class
        for i in range(len(self.cum_n_samp_per_cls)):
            self.cum_n_samp_per_cls[i] += (Tb == i).sum()

    def _update_weights(self, loss, Tb):
        ## Calculate Weights
        weights = torch.zeros_like(Tb, dtype=torch.float)
        probs_arr = 1. * self.cum_n_samp_per_cls / self.cum_n_samp_per_cls.sum()
        non_zero_mask = (probs_arr > 0)
        recip_probs_arr = torch.zeros_like(probs_arr)
        recip_probs_arr[non_zero_mask] = probs_arr[non_zero_mask] ** (-1)
        for i in range(self.n_classes_int):
            mask = (Tb == i)
            weights[mask] += recip_probs_arr[i]
        self.weights_norm = (weights / weights.mean()).to(loss.dtype).to(loss.device)

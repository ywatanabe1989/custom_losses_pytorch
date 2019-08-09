import torch

class MultiTaskLoss(torch.nn.Module):

  def __init__(self, is_regression):
    super(MultiTaskLoss, self).__init__()
    self.n_tasks = len(is_regression)
    self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks))

  def forward(self, losses):
    multi_task_losses = []
    stds = torch.exp(self.log_vars)**(1/2)
    coeffs = 1 / ( (is_regression+1)*(stds**2) )

    multi_task_losses = coeffs*losses + torch.log(stds)
    return muti_task_losses

'''
usage
is_regression = [True, True, False]
multitaskloss_instance = MultiTaskLoss(is_regression)
params = list(model.parameters()) + list(multitaskloss_instance.parameters())
torch.optim.Adam(params, lr=1e-3)

multitaskloss = multitaskloss_instance(losses)

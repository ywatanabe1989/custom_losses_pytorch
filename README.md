# custom_losses_pytorch
PInball Loss (a.k.a Quantile Loss)
balance_loss (function for tackling imbalanced data)

# Usage
from pinball_loss import PinballLoss

...

￼criterion = PinballLoss(quantile=0.05) # just like nn.MSELoss()

￼...

loss = criterion(output, target, reduction='sum')

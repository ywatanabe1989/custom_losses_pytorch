# custom_losses_pytorch
PInball Loss (a.k.a Quantile Loss)

# Usage
from pinball_loss import PinballLoss

...
￼criterion = PinballLoss(quantile=0.05) # just like nn.MSELoss()
￼...

loss = criterion(output, target, reduction='sum')

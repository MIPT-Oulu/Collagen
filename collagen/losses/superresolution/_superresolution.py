import numpy as np
from torch import nn, sigmoid, sum, max, log10, sqrt
from torch.nn import functional as F


class PSNRLoss(nn.Module):
    """
    Calculates the Peak signal-to-noise ratio and returns 100 - psnr as the loss value.
    """

    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, logits, targets):
        """Calculates Peak Signal-to-Noise ratio."""
        num = logits.size(0)
        preds = sigmoid(logits.view(num, -1))
        targets = targets.view(num, -1)
        mse = sum((targets - preds) ** 2).div(preds.size(1))
        #mse = np.mean((targets - preds) ** 2)
        if mse == 0:
            print('Identical images found!')
            return np.inf
        max_value = max(max(targets), max(preds))
        psnr = 20 * log10(max_value / sqrt(mse))
        return 100 - psnr

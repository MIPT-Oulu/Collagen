import numpy as np
from torch import nn, sigmoid


class PSNRLoss(nn.Module):
    """
    Calculates the Peak signal-to-noise ratio and returns 100 - psnr as the loss value.
    """

    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, logits, targets):
        """Calculates Peak Signal-to-Noise ratio."""
        num = logits.shape(0)
        preds = sigmoid(logits.view(num, -1))
        targets = targets.view(num, -1)
        mse = np.mean((targets - preds) ** 2)
        if mse == 0:
            print('Identical images found!')
            return np.inf
        max_value = max(np.max(targets), np.max(preds))
        psnr = 20 * np.log10(max_value / np.sqrt(mse))
        return 100 - psnr

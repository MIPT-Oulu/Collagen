import torch
from torch import nn


class BCEWithLogitsLoss2d(nn.Module):
    """Computationally stable version of 2D BCE loss.
    """

    def __init__(self):
        super(BCEWithLogitsLoss2d, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(None, reduction='mean')

    def forward(self, logits, targets):
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(logits_flat, targets_flat)


class SoftJaccardLoss(nn.Module):
    """SoftJaccard loss for binary problems.
    """

    def __init__(self, use_log=False):
        super(SoftJaccardLoss, self).__init__()
        self.use_log = use_log

    def forward(self, logits, labels):
        num = labels.size(0)
        m1 = torch.sigmoid(logits.view(num, -1))
        m2 = labels.view(num, -1)
        intersection = (m1 * m2).sum(1)
        score = (intersection + 1e-15) / (m1.sum(1) + m2.sum(1) - intersection + 1e-15)
        jaccard = score.sum(0) / num

        if not self.use_log:
            score = 1 - jaccard
        else:
            score = -torch.log(jaccard)
        return score

class SoftDiceLoss(nn.Module):
    """SoftJaccard loss for binary problems.
    """

    def forward(self, logits, labels):
        num = labels.size(0)
        m1 = torch.sigmoid(logits.view(num, -1))
        m2 = labels.view(num, -1)
        intersection = (m1 * m2).sum(1)
        score = (intersection + 1e-15) / (m1.sum(1) + m2.sum(1) + 1e-15)
        dice = score.sum(0) / num

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combination loss.
    Used to combine several existing losses, e.g. Dice and BCE
    """

    def __init__(self, losses, weights=None):
        super(CombinedLoss, self).__init__()
        self.losses = losses
        if weights is None:
            weights = [1 / len(losses)] * len(losses)

        self.weights = weights

    def forward(self, inputs, targets):
        loss = 0
        for l, w in zip(self.losses, self.weights):
            loss += l(inputs, targets) * w
        return loss


class BCEJaccardLoss(torch.nn.Module):
    def __init__(self, log_jaccard=True):
        super(BCEJaccardLoss, self).__init__()
        self.jaccard = SoftJaccardLoss(use_log=log_jaccard)
        self.bce = BCEWithLogitsLoss2d()

    def forward(self, logits, targets):
        bs = targets.size(0)
        use_jaccard = targets.view(bs, -1).sum(1).gt(0)

        loss = self.bce(logits, targets)
        if use_jaccard.sum() != 0:
            loss += self.jaccard(logits[use_jaccard], targets[use_jaccard])

        return loss

class BCEDiceLoss(torch.nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.dice = SoftDiceLoss()
        self.bce = BCEWithLogitsLoss2d()

    def forward(self, logits, targets):
        bs = targets.size(0)

        loss = self.bce(logits, targets)
        loss += self.dice(logits, targets)

        return loss

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

class MTLoss(nn.Module):
    def __init__(self, alpha_cls=0.4, alpha_st_cons=0.3, alpha_aug_cons=0.01):
        super().__init__()
        self.__loss_cls = CrossEntropyLoss(size_average=False)
        self.__alpha_cls = alpha_cls
        self.__alpha_st = alpha_st_cons
        self.__alpha_aug_cons = alpha_aug_cons
        self.consistency_rampup = 5.0
        self.__losses = {'loss': None, 'loss_aug_cons': None, 'loss_st_cons': None}
        self.__count = 0
        self.__consistency = 5 # 100

    def softmax_mse_loss(self, input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        num_classes = input_logits.size()[1]
        return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

    def symmetric_mse_loss(self, input1, input2):
        """Like F.mse_loss but sends gradients to both directions

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to both input1 and input2.
        """
        assert input1.size() == input2.size()
        num_classes = input1.size()[1]
        return torch.sum((input1 - input2) ** 2) / num_classes

    def get_current_consistency_weight(self):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.__consistency * sigmoid_rampup(self.__count, self.consistency_rampup)

    def forward(self, pred: torch.Tensor, target: torch.Tensor or tuple):
        minibatch_size = pred.shape[0]
        if target['name'] == 'l':
            target_cls = target['target'].type(torch.int64)
            st_logits = target['st_logits']
            te_logits = target['te_logits']
            loss_cls = self.__loss_cls(pred, target_cls) / minibatch_size
            if st_logits.shape[0] < 1:
                raise ValueError("Num of features must be at least 1, but found {}".format(st_logits.shape[0]))
            loss_aug_cons = self.softmax_mse_loss(st_logits[0, :, :], st_logits[1, :, :]) / minibatch_size
            loss_st_cons = self.symmetric_mse_loss(te_logits, pred) / minibatch_size

            self.__losses['loss_cls'] = loss_cls
            self.__losses['loss_st_cons'] = self.get_current_consistency_weight() * loss_st_cons
            self.__losses['loss_aug_cons'] = self.__alpha_aug_cons * loss_aug_cons
            _loss = self.__losses['loss_cls'] + self.__losses['loss_st_cons'] + self.__losses['loss_st_cons']
            self.__losses['loss'] = _loss

        elif target['name'] == 'u':
            st_logits = target['st_logits']
            te_logits = target['te_logits']
            if st_logits.shape[0] < 1:
                raise ValueError("Num of features must be at least 1, but found {}".format(st_logits.shape[0]))
            loss_aug_cons = self.softmax_mse_loss(st_logits[0, :, :], st_logits[1, :, :]) / minibatch_size
            loss_st_cons = self.symmetric_mse_loss(te_logits, pred) / minibatch_size

            self.__losses['loss_cls'] = None
            self.__losses['loss_st_cons'] = self.get_current_consistency_weight() * loss_st_cons
            self.__losses['loss_aug_cons'] = self.__alpha_aug_cons* loss_aug_cons
            _loss = self.__losses['loss_st_cons'] + self.__losses['loss_st_cons']
            self.__losses['loss'] = _loss
        else:
            raise ValueError("Target length is 1 or 3, but found {}".format(len(target)))

        self.__count += 1
        return _loss

    def get_loss_by_name(self, name):
        if name in self.__losses:
            return self.__losses[name]
        else:
            return None

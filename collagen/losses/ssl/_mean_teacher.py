import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from collagen.callbacks.lrscheduler.utils.ramps import sigmoid_rampup


class MTLoss(nn.Module):
    def __init__(self, alpha_cls=1.0, consistency=10, consistency_rampup=5, logit_distance_cost=0.01):
        super().__init__()
        self.__loss_cls = CrossEntropyLoss(size_average=False)
        self.__alpha_cls = alpha_cls
        self.__logit_distance_cost = logit_distance_cost
        self.consistency_rampup = consistency_rampup
        self.__losses = {'loss': None, 'loss_aug_cons': None, 'loss_s_t_cons': None}
        self.__count = 0
        self.__consistency = consistency  # 5 # 100

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
        n_minibatches = 2
        if target['name'] == 'l_st':
            target_cls = target['target'].type(torch.int64)
            st_logits = target['logits']

            loss_aug_cons = self.symmetric_mse_loss(st_logits, pred) / (minibatch_size * n_minibatches)
            loss_cls = self.__loss_cls(pred, target_cls) / (minibatch_size * n_minibatches)

            self.__losses['loss_cls'] = loss_cls
            self.__losses['loss_s_t_cons'] = None
            self.__losses['loss_aug_cons'] = self.__logit_distance_cost * loss_aug_cons
            _loss = (self.__losses['loss_cls'] + self.__losses['loss_aug_cons'])
        elif target['name'] == 'u_st':
            st_logits = target['logits']

            loss_aug_cons = self.symmetric_mse_loss(st_logits, pred) / (minibatch_size * n_minibatches)

            self.__losses['loss_cls'] = None
            self.__losses['loss_s_t_cons'] = None
            self.__losses['loss_aug_cons'] = self.__logit_distance_cost * loss_aug_cons
            _loss = self.__losses['loss_aug_cons']
        elif target['name'] == 'l_te':
            te_logits = target['logits']
            target_cls = target['target'].type(torch.int64)

            loss_cls = self.__loss_cls(pred, target_cls) / (minibatch_size * n_minibatches)
            loss_st_cons = self.softmax_mse_loss(te_logits, pred) / (minibatch_size * n_minibatches)

            self.__losses['loss_cls'] = loss_cls
            self.__losses['loss_s_t_cons'] = self.get_current_consistency_weight() * loss_st_cons
            self.__losses['loss_aug_cons'] = None
            _loss = self.__losses['loss_s_t_cons'] + self.__losses['loss_cls']

        elif target['name'] == 'u_te':
            te_logits = target['logits']

            loss_st_cons = self.softmax_mse_loss(te_logits, pred) / (minibatch_size * n_minibatches)

            self.__losses['loss_cls'] = None
            self.__losses['loss_s_t_cons'] = self.get_current_consistency_weight() * loss_st_cons
            self.__losses['loss_aug_cons'] = None
            _loss = self.__losses['loss_s_t_cons']
        elif target['name'] == 'l_te_eval':
            target_cls = target['target'].type(torch.int64)

            loss_cls = self.__loss_cls(pred, target_cls) / minibatch_size

            self.__losses['loss_cls'] = loss_cls
            self.__losses['loss_s_t_cons'] = None
            self.__losses['loss_aug_cons'] = None
            _loss = self.__losses['loss_cls']
        else:
            raise ValueError("Target length is 1 or 3, but found {}".format(len(target)))

        self.__losses['loss'] = _loss
        self.__count += 1
        return _loss

    def get_loss_by_name(self, name):
        if name in self.__losses:
            return self.__losses[name]
        else:
            return None

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.nn import functional as F
from torch import Tensor


class SemixupLoss(Module):
    def __init__(self, in_manifold_coef=2.0, in_out_manifold_coef=2.0, ic_coef=4.0, cons_mode='mse'):
        super().__init__()
        self._cons_mode = cons_mode
        if cons_mode == 'mse':
            self._loss_cons = self.softmax_mse_loss
        elif cons_mode == 'kl':
            self._loss_cons = self.softmax_kl_loss

        self._loss_cls = CrossEntropyLoss(reduction='sum')

        self._in_manifold_coef = in_manifold_coef
        self._losses = {'loss_cls': None, 'loss_in_mnf': None}

        self._n_minibatches = 1.0
        self._ic_coef = ic_coef
        self._in_out_manifold_coef = in_out_manifold_coef

        self._n_aug = 1

        if self._n_aug < 1:
            raise ValueError('Not support {} augmentations'.format(self._n_aug))

    @staticmethod
    def softmax_kl_loss(input_logits, target_logits):
        """Takes softmax on both sides and returns KL divergence

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        n_classes = target_logits.shape[1]
        return F.kl_div(input_log_softmax, target_softmax, reduction='sum') / n_classes

    @staticmethod
    def softmax_mse_loss(input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        n_classes = input_logits.size()[1]
        return F.mse_loss(input_softmax, target_softmax, reduction='sum') / n_classes

    def forward(self, pred: Tensor, target: Tensor or dict):
        n_minibatch_size = pred.shape[0]
        if target['name'] == 'u_mixup':
            mixup_logits = pred

            if len(target['logits_mixup'].shape) == 2:
                loss_ic = self._loss_cons(mixup_logits, target['logits_mixup'])
            else:
                raise ValueError('Not support augmented logits with shape {}'.format(target['logits_mixup'].shape))

            loss_cons_aug = self._loss_cons(target['logits_aug'], target['logits'])
            loss_cons_aug_mixup = self._loss_cons(mixup_logits, target['logits_aug'])
            loss_cons_mixup = self._loss_cons(target['logits'], mixup_logits)

            self._losses['loss_in_mnf'] = self._in_manifold_coef * loss_cons_aug / (
                    self._n_minibatches * n_minibatch_size)
            self._losses['loss_ic'] = self._ic_coef * loss_ic / (self._n_minibatches * n_minibatch_size)
            self._losses['loss_inout_mnf'] = self._in_out_manifold_coef * (loss_cons_aug_mixup + loss_cons_mixup) / (
                    self._n_minibatches * n_minibatch_size)
            self._losses['loss'] = self._losses['loss_inout_mnf'] + self._losses['loss_in_mnf'] + self._losses['loss_ic']
            self._losses['loss_cls'] = None
        elif target['name'] == 'l_mixup':
            target_cls1 = target['target'].type(torch.int64)
            target_cls2 = target['target_bg'].type(torch.int64)
            alpha = target['alpha']

            # DEBUG
            # loss_cls = alpha * self._loss_cls(pred, target_cls1) + (1 - alpha) * self._loss_cls(pred, target_cls2)
            loss_cls = self._loss_cls(pred, target_cls1)

            self._losses['loss_inout_mnf'] = None
            self._losses['loss_in_mnf'] = None
            self._losses['loss_ic'] = None
            self._losses['loss_cls'] = loss_cls / (self._n_minibatches * n_minibatch_size)
            self._losses['loss'] = self._losses['loss_cls']
        elif target['name'] == 'l_norm':
            target_cls = target['target'].type(torch.int64)
            loss_cls = self._loss_cls(pred, target_cls)

            self._losses['loss_inout_mnf'] = None
            self._losses['loss_in_mnf'] = None
            self._losses['loss_ic'] = None
            self._losses['loss_cls'] = loss_cls / (self._n_minibatches * n_minibatch_size)
            self._losses['loss'] = self._losses['loss_cls']
        else:
            raise ValueError("Not support target name {}".format(target['name']))

        _loss = self._losses['loss']
        return _loss

    def get_loss_by_name(self, name):
        if name in self._losses:
            return self._losses[name]
        else:
            return None

from collagen.core import Module
from collagen.core import Callback
import torch.nn as nn
import torch

"""

class DCGAN(Module):
    def __init__(self, device, train_mode=True, with_backward=True, accumulate_grad=False, nc=1, nz=100, lr=1e-3, weight_decay=1e-1, unrolling_steps=1):
        super(DCGAN, self).__init__()

        self.__nc = nc
        self.__nz = nz
        self.__device = device

        self.__real_label = 1.0
        self.__fake_label = 0.0

        self.__unrolling_steps = unrolling_steps
        self.__modules = nn.ModuleDict()
        self.__modules["D"] = Discriminator(ngpu=1, nc=self.__nc, ndf=32)
        self.__modules["G"] = Generator(ngpu=1, nc=self.__nc, ngf=32)

        self.__losses = nn.ModuleDict()
        self.__losses["D"] = nn.BCELoss()

        self.__optimizers = {"D": torch.,
                            "G": torch.optim.Adam(params=self.__modules["G"].parameters(), lr=lr,
                                       weight_decay=weight_decay)}

        self.__train_mode = train_mode
        self.__with_backward = with_backward
        self.__accumulate_grad = accumulate_grad

        self.optimize_cb = self.OptimizerCallback(self.__optimizers, self.__modules, self.__losses,
                                                  self.__device, accumulate_grad=self.__accumulate_grad,
                                                  nz=self.__nz)

    def get_module_by_name(self, name):
        if name in self.__modules:
            return self.__modules[name]
        else:
            raise ValueError('Module {} not found'.format(name))

    def forward(self, input):
        return self.__modules["D"](input)

    def run(self, input, labels=None, with_backward: bool or None = None, accumulate_grade: bool or None = None):
        if input is None:
            raise ValueError("Input must be not None.")

        batch_size = input.shape[0]

        if accumulate_grade is None:
            accumulate_grade = self.__accumulate_grad

        if with_backward is None:
            with_backward = self.__with_backward

        if not accumulate_grade:
            self.__modules["G"].zero_grad()
            self.__modules["D"].zero_grad()

        if labels is None:
            labels = torch.full((batch_size,), self.__real_label, device=self.__device)

        # Real data
        preds = self.__modules["D"](input)
        err_d = self.__losses["D"](preds, labels)
        if with_backward:
            err_d.backward()

        return preds, err_d

    class OptimizerCallback(Callback):
        def __init__(self, optimizers, modules, losses, device, accumulate_grad=False, real_label=1.0, fake_label=0.0, nz=100):
            self.__modules = modules
            self.__losses = losses
            self.__accumulate_grad = accumulate_grad
            self.__optimizers = optimizers
            self.__device = device
            self.__real_label = real_label
            self.__fake_label = fake_label
            self.__nz = nz

        def on_optimize(self, input, target=None, accumulate_grad=None, **kwargs):
            batch_size = input.shape[0]

            if accumulate_grad is None:
                accumulate_grad = self.__accumulate_grad

            if not accumulate_grad:
                self.__modules["G"].zero_grad()
                self.__modules["D"].zero_grad()

            if target is not None:
                real_labels = target.float()
            else:
                real_labels = torch.full((batch_size,), self.__real_label, device=self.__device)

            fake_labels = real_labels.fill_(self.__fake_label)

            # Real data
            real_preds = self.__modules["D"](input)

            real_err_d = self.__losses["D"](real_preds, real_labels)
            real_err_d.backward()
            self.__optimizers["D"].step()

            # Fake data
            noise = torch.randn(batch_size, self.__nz, 1, 1, device=self.__device)
            fake_input = self.__modules["G"](noise)
            fake_preds = self.__modules["D"](fake_input)
            fake_err_d = self.__losses["D"](fake_preds, fake_labels)
            fake_err_d.backward()
            self.__optimizers["G"].step()

            return {"output_D": real_preds, "output_G": fake_input, "output_D_G": fake_preds}, \
                   {"loss_D": real_err_d, "loss_G": fake_err_d}
"""




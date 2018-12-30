from Collagen.collagen.core import Module
from Collagen.collagen.core import Callback
import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

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

        self.__optimizers = {"D": torch.optim.Adam(params=self.__modules["D"].parameters(), lr=lr,
                                       weight_decay=weight_decay),
                            "G": torch.optim.Adam(params=self.__modules["G"].parameters(), lr=lr,
                                       weight_decay=weight_decay)}

        self.__train_mode = train_mode
        self.__with_backward = with_backward
        self.__accumulate_grad = accumulate_grad

        self.optimize_cb = self.OptimizerCallback(self.__optimizers, self.__modules, self.__losses, self.__accumulate_grad, self.__device)

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
        def __init__(self, optimizers, modules, losses, device, accumulate_grad=False):
            self.__modules = modules
            self.__losses = losses
            self.__accumulate_grad = accumulate_grad
            self.__optimizers = optimizers
            self.__device = device

        def on_optimize(self, *args, **kwargs):
            if len(args) > 1:
                input = args[0]
                real_labels = args[1]
            elif "input" in kwargs and "label" in kwargs:
                input = kwargs["input"]
                real_labels = kwargs["label"]
            else:
                raise ValueError("Must have a least 1 input.")

            if input is None:
                raise ValueError("Input must be not None.")

            batch_size = input.shape[0]

            if not self.__accumulate_grad:
                self.__modules["G"].zero_grad()
                self.__modules["D"].zero_grad()

            real_labels = torch.full((batch_size,), real_labels, device=self.__device)
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






from torch import optim

from collagen.core import Stepper

from .fixtures import *


def test_stepper_single_batch_step_simple(dumb_net, classification_minibatch_two_class):
    net = dumb_net(16, 1)
    batch, target = classification_minibatch_two_class
    optimizer = optim.Adam(net.parameters(), lr=5e-2)
    criterion = nn.BCEWithLogitsLoss()
    torch.manual_seed(42)
    session = Stepper(net, optimizer, criterion)
    loss = 10000
    for i in range(50):
        loss = session.train_step(batch, target, return_out=True)[0]

    assert loss < 1e-1


def test_stepper_train_eval(dumb_net, classification_minibatches_seq_two_class):
    net = dumb_net(16, 1)
    optimizer = optim.Adam(net.parameters(), lr=5e-2)
    criterion = nn.BCEWithLogitsLoss()

    session = Stepper(net, optimizer, criterion)
    loss = 100000
    torch.manual_seed(42)
    val_losses = []
    for i in range(50):
        for batch, target in classification_minibatches_seq_two_class[:1]:
            loss = session.train_step(batch, target, return_out=True)[0]

        val_loss = 0
        for batch, target in classification_minibatches_seq_two_class[1:]:
            val_loss += session.eval_step(batch, target, return_out=True)[0]
        val_loss /= len(classification_minibatches_seq_two_class[1:])
        val_losses.append(val_loss)

    assert loss < 1e-1
    assert min(val_losses) * 4 < val_losses[0]

import pytest
import numpy as np
from .fixtures import classification_problem_output
from collagen.callbacks.metrics import ConfusionMeter
from sklearn.metrics import confusion_matrix


@pytest.mark.parametrize('batch_size', [5, 10, 32])
def test_confusion_meter(classification_problem_output, batch_size):
    gt, preds, n_classes = classification_problem_output
    cm = confusion_matrix(gt.numpy(), preds.argmax(1).numpy())

    meter = ConfusionMeter(n_classes)

    size = gt.size(0)
    n_batches = size // batch_size
    for i in range(n_batches+1):
        batch_gt = gt[i*batch_size:i*batch_size+batch_size]
        batch_preds = preds[i * batch_size:i * batch_size + batch_size, :]

        meter.on_minibatch_end(batch_gt, batch_preds)

    np.testing.assert_equal(cm, meter.current())

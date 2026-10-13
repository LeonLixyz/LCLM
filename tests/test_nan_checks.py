from types import SimpleNamespace

import pytest
import torch

from utils.nan_checks import has_non_finite_loss_and_gradients


@pytest.mark.parametrize('loss_bad,grad_bad', [(False, False), (True, False), (False, True)])
def test_finite_loss_does_not_hide_bad_gradients(loss_bad, grad_bad):
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, float('inf') if grad_bad else 1.0)
    calls = []
    def reduce(flag, reduction):
        calls.append(reduction)
        return flag
    accelerator = SimpleNamespace(device=torch.device('cpu'), reduce=reduce)
    result = has_non_finite_loss_and_gradients(
        loss=torch.tensor(float('nan') if loss_bad else 1.0), model=model, accelerator=accelerator)
    assert result == (loss_bad or grad_bad)
    assert calls == ['sum']


def test_remote_failure_is_respected_and_collective_failure_is_not_hidden():
    model = torch.nn.Linear(2, 1)
    accelerator = SimpleNamespace(device=torch.device('cpu'), reduce=lambda value, **kw: value + 1)
    assert has_non_finite_loss_and_gradients(loss=torch.tensor(1.0), model=model, accelerator=accelerator)
    def fail(*args, **kwargs):
        raise RuntimeError('collective failure')
    accelerator.reduce = fail
    with pytest.raises(RuntimeError, match='collective failure'):
        has_non_finite_loss_and_gradients(loss=torch.tensor(1.0), model=model, accelerator=accelerator)

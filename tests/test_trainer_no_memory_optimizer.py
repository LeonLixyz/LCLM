import torch
from torch.optim import AdamW

from train.optimizer_utils import clear_optimizer_group_grads, global_has_memory


def make_optimizer():
    decoder = torch.nn.Parameter(torch.tensor(1.0))
    encoder = torch.nn.Parameter(torch.tensor(1.0))
    adapter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = AdamW(
        [
            {"params": [decoder], "lr": 0.1, "weight_decay": 0.1},
            {"params": [encoder], "lr": 0.1, "weight_decay": 0.1},
            {"params": [adapter], "lr": 0.1, "weight_decay": 0.1},
        ]
    )
    return optimizer, decoder, encoder, adapter


def test_globally_uncompressed_step_does_not_advance_encoder_or_adapter_adamw():
    optimizer, decoder, encoder, adapter = make_optimizer()

    # Seed AdamW momentum/state with a compressed step first.
    (decoder + encoder + adapter).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)

    encoder_before = encoder.detach().clone()
    adapter_before = adapter.detach().clone()
    decoder_before = decoder.detach().clone()

    # A connected dummy forward gives encoder/adapter explicit zero grads. Clear
    # those groups to None before AdamW.step so momentum and decay do not move.
    (decoder + 0.0 * encoder + 0.0 * adapter).backward()
    assert not global_has_memory(False, device=torch.device("cpu"), distributed=False)
    clear_optimizer_group_grads(optimizer, {1, 2})
    optimizer.step()

    assert not torch.equal(decoder, decoder_before)
    assert torch.equal(encoder, encoder_before)
    assert torch.equal(adapter, adapter_before)


def test_local_memory_flag_is_true_without_distributed_collective():
    assert global_has_memory(True, device=torch.device("cpu"), distributed=False)


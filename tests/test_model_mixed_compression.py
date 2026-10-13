from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from latent_context.encoder import Encoder
from latent_context.model import LCLM


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.pad_token_id = 7
        self.calls = []

    def forward(self, segments):
        self.calls.append([[int(token) for token in segment] for segment in segments])
        return [
            self.scale
            * torch.tensor(
                [[float(sum(segment)), float(len(segment))]],
                device=self.scale.device,
            )
            for segment in segments
        ]


class RecordingAdapter(nn.Module):
    adapter_type = "mlp"

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        nn.init.eye_(self.fc1.weight)
        self.calls = 0

    def forward(self, embeddings):
        self.calls += 1
        return self.fc1(embeddings)


def make_lclm_stub():
    model = object.__new__(LCLM)
    nn.Module.__init__(model)
    model.encoder = RecordingEncoder()
    model.adapter = RecordingAdapter()
    model.accelerator = None
    return model


@pytest.mark.parametrize(
    "memory_token_ids, expected",
    [
        ([[], [[11, 12]]], [[], [[11, 12]]]),
        ([[[11, 12]], []], [[[11, 12]], []]),
        ([[11, 12], [13]], [[[11, 12]], [[13]]]),
    ],
)
def test_normalize_mixed_batch_does_not_depend_on_first_sample(
    memory_token_ids, expected
):
    assert LCLM._normalize_batched_memory_token_ids(memory_token_ids, 2) == expected


def test_normalize_and_validate_mixed_memory_metadata():
    normalized = LCLM._normalize_and_validate_memory_batch(
        memory_token_ids=[[], [[11, 12, 13]]],
        memory_positions=[[], [(4, 6)]],
        latent_counts=[[], [2]],
        batch_size=2,
        sequence_length=8,
    )

    assert normalized == (
        [[], [[11, 12, 13]]],
        [[], [(4, 6)]],
        [[], [2]],
    )


def test_inconsistent_uncompressed_metadata_is_rejected():
    with pytest.raises(ValueError, match="inconsistent memory metadata"):
        LCLM._normalize_and_validate_memory_batch(
            memory_token_ids=[[], [[11, 12]]],
            memory_positions=[[], []],
            latent_counts=[[], [1]],
            batch_size=2,
        )


def test_mixed_batch_is_flattened_into_one_encoder_and_adapter_call():
    model = make_lclm_stub()

    embeddings, zero_dependency = model._process_batched_latent_embeddings(
        [[], [[1, 2], [3]], [[4, 5]]],
        ensure_participation=False,
    )

    assert model.encoder.calls == [[[1, 2], [3], [4, 5]]]
    assert model.adapter.calls == 1
    assert [len(sample) for sample in embeddings] == [0, 2, 1]
    assert torch.equal(embeddings[1][0], torch.tensor([[3.0, 2.0]]))
    assert torch.equal(embeddings[2][0], torch.tensor([[9.0, 2.0]]))
    assert zero_dependency is None


def test_legacy_latent_helper_handles_leading_uncompressed_sample():
    model = make_lclm_stub()

    embeddings = model._process_latent_embeddings([[], [[1, 2]]])

    assert [len(sample) for sample in embeddings] == [0, 1]
    assert model.encoder.calls == [[[1, 2]]]
    assert model.adapter.calls == 1


def test_all_empty_batch_uses_connected_zero_gradient_dummy():
    model = make_lclm_stub()
    model.train()

    embeddings, zero_dependency = model._process_batched_latent_embeddings(
        [[], []],
        ensure_participation=True,
    )

    assert embeddings == [[], []]
    assert model.encoder.calls == [[[model.encoder.pad_token_id]]]
    assert model.adapter.calls == 1
    assert zero_dependency is not None
    assert zero_dependency.item() == 0.0
    assert zero_dependency.requires_grad

    decoder_embeds = torch.randn(2, 3, 2, requires_grad=True)
    (decoder_embeds + zero_dependency).square().sum().backward()

    assert decoder_embeds.grad is not None
    assert model.encoder.scale.grad is not None
    assert torch.count_nonzero(model.encoder.scale.grad) == 0
    assert model.adapter.fc1.weight.grad is not None
    assert torch.count_nonzero(model.adapter.fc1.weight.grad) == 0


def test_encoder_sync_dummy_forwards_are_connected_to_backward(monkeypatch):
    encoder = object.__new__(Encoder)
    nn.Module.__init__(encoder)
    encoder.max_encode_batch_size = 1
    encoder.accelerator = SimpleNamespace(num_processes=2)
    encoder.pad_token_id = 0
    encoder.weight = nn.Parameter(torch.tensor(1.0))
    forward_outputs = []

    def fake_forward_one(self, input_ids, attention_mask, position_ids):
        del attention_mask, position_ids
        output = self.weight * torch.ones(
            (*input_ids.shape, 1), dtype=torch.float32, device=input_ids.device
        )
        output.retain_grad()
        forward_outputs.append(output)
        return output

    def fake_all_reduce(value, op):
        del op
        value.fill_(3)

    encoder._forward_one = MethodType(fake_forward_one, encoder)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    output = encoder._batched_encode(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=torch.arange(2).unsqueeze(0),
        total=1,
        device=input_ids.device,
    )
    output.sum().backward()

    assert len(forward_outputs) == 3
    assert torch.equal(forward_outputs[0].grad, torch.ones_like(forward_outputs[0]))
    for dummy_output in forward_outputs[1:]:
        assert dummy_output.grad is not None
        assert torch.count_nonzero(dummy_output.grad) == 0


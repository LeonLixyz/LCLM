"""CPU checks for inference/optimized_encoder.py.

CUDA graphs and torch.compile need a GPU; these tests cover everything else
(right-padded packing, mask-free causal attention, pooling, install/restore,
argument validation) against the real ``Encoder`` with a tiny Qwen3 model.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3Model

from inference import optimized_encoder as oe
from latent_context.encoder import Encoder

PAD = 0


def tiny_encoder(**overrides):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attn_implementation="sdpa",
    )
    kwargs = dict(
        compression_ratio=4,
        max_length=256,
        encoder_model=Qwen3Model(config).eval(),
        encoder_tokenizer=SimpleNamespace(pad_token_id=PAD),
        train_encoder=False,
        max_encode_batch_size=0,
        pooling="mean",
        encoder_mask_type="causal",
        encoder_window_size=16,
        boundary_overlap=0,
    )
    kwargs.update(overrides)
    return Encoder(**kwargs)


class TinyLCLM(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder


class CpuForward:
    """Stand-in for the CUDA-graph forward: same call, eager on CPU."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.graphs = {}
        self.capture_log = []
        self.shapes = []

    def __call__(self, input_ids, attention_mask, position_ids):
        self.shapes.append(tuple(input_ids.shape))
        with torch.no_grad():
            return oe._encode_without_mask(self.encoder.embed_model, input_ids, position_ids)


SAMPLES = [
    list(range(1, 38)),  # 3 windows: 16, 16, 5 (partial window, tail group of 1)
    [],  # empty sample
    list(range(5, 21)),  # exactly one window
    [7, 8, 9],  # shorter than one group
]


def install_on_cpu(encoder, batch_windows):
    optimization = oe.EncoderOptimization(encoder, compile_encoder=False, batch_windows=batch_windows)
    optimization._graphed = CpuForward(encoder)
    optimization._install()
    return optimization


@pytest.mark.parametrize("batch_windows", [1, 2, 32])
def test_right_padded_path_matches_stock_encoder(batch_windows):
    encoder = tiny_encoder()
    with torch.no_grad():
        expected = encoder(SAMPLES)
    optimization = install_on_cpu(encoder, batch_windows)
    with torch.no_grad():
        actual = encoder(SAMPLES)

    assert [t.shape for t in actual] == [t.shape for t in expected]
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)

    windows = 3 + 1 + 1
    expected_batches = -(-windows // batch_windows)
    assert len(optimization._graphed.shapes) == expected_batches
    assert all(shape[0] <= batch_windows for shape in optimization._graphed.shapes)


def test_restore_reinstates_stock_encoder():
    encoder = tiny_encoder(max_encode_batch_size=7)
    encoder.embed_model.config.use_cache = True
    optimization = install_on_cpu(encoder, 32)
    assert encoder.max_encode_batch_size == 32
    assert encoder.embed_model.config.use_cache is False
    assert "forward" in encoder.__dict__ and "_forward_one" in encoder.__dict__

    optimization.restore()
    assert encoder.max_encode_batch_size == 7
    assert encoder.embed_model.config.use_cache is True
    assert "forward" not in encoder.__dict__ and "_forward_one" not in encoder.__dict__
    assert getattr(encoder, "_lclm_optimization", None) is None
    with torch.no_grad():
        encoder(SAMPLES)


@pytest.mark.parametrize("pool_len", [1, 3, 4, 5, 8, 15, 16])
def test_mean_pool_matches_groupwise(pool_len):
    encoder = tiny_encoder()
    hs = torch.randn(pool_len, 32)
    torch.testing.assert_close(
        oe._mean_pool(encoder, hs, pool_len, 32),
        encoder._pool_groupwise(hs, pool_len, 32),
    )


def test_refuses_training_mode():
    model = TinyLCLM(tiny_encoder()).train()
    with pytest.raises(ValueError, match="eval"):
        oe.enable_encoder_optimization(model)
    assert "forward" not in model.encoder.__dict__


def test_requires_cuda():
    model = TinyLCLM(tiny_encoder()).eval()
    with pytest.raises(ValueError, match="CUDA"):
        oe.enable_encoder_optimization(model)


def test_rejects_bad_batch_windows():
    model = TinyLCLM(tiny_encoder()).eval()
    with pytest.raises(ValueError, match="batch_windows"):
        oe.enable_encoder_optimization(model, batch_windows=0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(encoder_mask_type="bidirectional"), "causal"),
        (dict(boundary_overlap=4), "boundary_overlap"),
        (dict(pooling="eos"), "pooling"),
        (dict(pooling="concat"), "pooling"),
    ],
)
def test_rejects_unsupported_encoders(overrides, message):
    model = TinyLCLM(tiny_encoder(**overrides)).eval()
    with pytest.raises(ValueError, match=message):
        oe.enable_encoder_optimization(model)

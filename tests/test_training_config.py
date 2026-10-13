"""Checkpoint semantics and the batch size passed to DeepSpeed."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from train.launch_train import ModelArguments, DataArguments, TrainingConfig
from train.trainer import LCLMTrainer
from utils.checkpointing import validate_resume_model_config


def legacy_config():
    return dict(pooling_token="summary_mean", chunk_size=16,
                batch_summary_tokens=1024, embed_mask_type="causal",
                overlap_tokens=0, use_adapter_attention=False)


def write_config(path, config):
    (path / "model_config.json").write_text(json.dumps(config))


def test_summary_mean_preserves_windowed_compression(tmp_path):
    write_config(tmp_path, legacy_config())
    saved = validate_resume_model_config(tmp_path, ModelArguments(), 16)
    assert saved == dict(pooling="mean", compression_ratio=16,
                         encoder_window_size=1024, encoder_mask_type="causal",
                         boundary_overlap=0, adapter_type="mlp")


@pytest.mark.parametrize("field,value,reason", [
    ("chunk_size", 8, "compression_ratio"),
    ("batch_summary_tokens", 256, "encoder_window_size"),
    ("embed_mask_type", "bidirectional", "encoder_mask_type"),
    ("overlap_tokens", 32, "boundary_overlap"),
    ("overlap_tokens", None, "boundary_overlap"),
    ("pooling_token", "mean", "encoder_window_size"),
    ("pooling_token", "summary_concat", "pooling"),
])
def test_incompatible_legacy_encoder_is_rejected(tmp_path, field, value, reason):
    config = legacy_config()
    config[field] = value
    write_config(tmp_path, config)
    with pytest.raises(ValueError, match=reason):
        validate_resume_model_config(tmp_path, ModelArguments(), 16)


def test_incompatible_legacy_adapter_is_rejected(tmp_path):
    config = dict(legacy_config(), use_adapter_attention=True,
                  adapter_order="attention_mlp", num_adapter_layers=2)
    write_config(tmp_path, config)
    with pytest.raises(ValueError, match="adapter_type"):
        validate_resume_model_config(tmp_path, ModelArguments(), 16)


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_loads_encoder_decoder_and_adapter_weights(tmp_path, monkeypatch, legacy):
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
    from latent_context.adapter import Adapter

    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=16, pad_token_id=0)
    decoder, encoder = Qwen3ForCausalLM(config), Qwen3Model(config)
    adapter = Adapter(32, 32)
    decoder_dir, encoder_dir, adapter_dir, filename = (
        ("llm", "embedder", "projectors", "code_adapter.safetensors") if legacy
        else ("decoder", "encoder", "adapter", "adapter.safetensors")
    )
    decoder.save_pretrained(tmp_path / decoder_dir)
    encoder.save_pretrained(tmp_path / encoder_dir)
    (tmp_path / adapter_dir).mkdir()
    save_file(adapter.state_dict(), tmp_path / adapter_dir / filename)
    saved = legacy_config() if legacy else dict(
        pooling="mean", compression_ratio=16, encoder_window_size=1024,
        encoder_mask_type="causal", boundary_overlap=0, adapter_type="mlp")
    write_config(tmp_path, saved)
    special = {"<|memory_start|>": 120, "<|memory_end|>": 121, "<|memory|>": 122}
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
        pad_token="pad", eos_token="eos", convert_tokens_to_ids=special.__getitem__)
    monkeypatch.setattr("train.trainer.AutoTokenizer.from_pretrained", lambda *a, **k: tokenizer)
    trainer = object.__new__(LCLMTrainer)
    trainer.model_args = ModelArguments(train_encoder=True,
        decoder_attn_implementation="eager", embed_attn_implementation="eager")
    trainer.data_args = DataArguments(use_packing=False)
    trainer.training_args = TrainingConfig(compression_ratio=16, use_liger_kernel=False,
        encoder_gradient_checkpointing=False, decoder_gradient_checkpointing=False)
    trainer._resume_from_checkpoint(str(tmp_path))
    for loaded, original in ((trainer.model.decoder, decoder),
                             (trainer.model.encoder.embed_model, encoder),
                             (trainer.model.adapter, adapter)):
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
    assert trainer.model.encoder.pooling == "mean"
    assert trainer.model.encoder.encoder_window_size == 1024
    assert trainer.model.encoder.is_causal
    (tmp_path / adapter_dir / filename).unlink()
    with pytest.raises(FileNotFoundError, match="adapter weights"):
        trainer._resume_from_checkpoint(str(tmp_path))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_deepspeed_uses_actual_iterable_loader_batch_size(batch_size):
    class Rows(IterableDataset):
        def __iter__(self):
            return iter(range(8))

    class Prepared(Exception):
        pass

    config = {"train_micro_batch_size_per_gpu": "auto"}
    trainer = object.__new__(LCLMTrainer)
    trainer.train_dataloader = DataLoader(Rows(), batch_size=batch_size)
    trainer.model = trainer.optimizer = trainer.scheduler = None

    def prepare(*args):
        assert config["train_micro_batch_size_per_gpu"] == next(iter(trainer.train_dataloader)).shape[0]
        raise Prepared

    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=1, deepspeed_config=config)),
        prepare=prepare)
    with pytest.raises(Prepared):
        trainer.prepare_for_training()

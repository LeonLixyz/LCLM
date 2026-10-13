"""Target-side memory must be causal, differentiable, and isolated across packs."""
import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from data.dynamic_packing_dataset import DynamicPackedDataset
from data.packing_utils import collate_packed_batch
from latent_context.model import LCLM
from latent_context.processor import LCLMProcessor


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    special = {'<|memory_start|>': 120, '<|memory_end|>': 121, '<|memory|>': 122}

    def convert_tokens_to_ids(self, token):
        return self.special[token]

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 64 + 1 for char in text]


def cot_batch(text):
    tokenizer = Tokenizer()
    loader = object.__new__(DynamicPackedDataset)
    loader.embed_tokenizer = tokenizer
    loader.compression_ratio = 16
    loader.memory_start_id, loader.memory_end_id, loader.memory_id = 120, 121, 122
    target = {
        'base_input_ids': [5, 6, 7, 8, 120, 122, 121, 9, 10, 11],
        'base_labels': [-100, -100, 7, 8, -100, -100, -100, 9, 10, 11],
        'memory_strings': [text], 'memory_positions': [4],
    }
    other = {
        'base_input_ids': [31, 32, 33, 34],
        'base_labels': [-100, 32, 33, 34],
        'memory_strings': [], 'memory_positions': [],
    }
    examples = [loader._expand_example(ex) for ex in (target, other)]
    return collate_packed_batch(examples, target_length=16, pad_token_id=0)


def model_on_gpu(checkpointing):
    tokenizer = Tokenizer()
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, attention_dropout=0.0, pad_token_id=0, use_cache=False)
    config._attn_implementation = 'sdpa'
    decoder = Qwen3ForCausalLM(config)
    encoder_config = copy.deepcopy(config)
    encoder_config._attn_implementation = 'flash_attention_2'
    encoder = Qwen3Model(encoder_config)
    processor = LCLMProcessor(tokenizer, tokenizer, compression_ratio=16)
    model = LCLM(decoder, tokenizer, encoder, tokenizer, processor,
        compression_ratio=16, encoder_window_size=1024, encoder_mask_type='bidirectional',
        packed_attention_backend='flash', adapter_type='mlp', use_fused_ce=False)
    model._enable_packed_attention('flash')
    model = model.to(device='cuda', dtype=torch.bfloat16).train()
    if checkpointing:
        for part in (model.decoder, model.encoder.embed_model):
            part.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU CoT compression check')
@pytest.mark.parametrize('checkpointing', [False, True])
def test_cot_is_encoded_online_without_future_or_cross_document_leakage(checkpointing):
    from latent_context.packed_flash import clear_packed_sample_lens
    torch.manual_seed(20260911)
    model = model_on_gpu(checkpointing)
    original, changed = cot_batch('ab' * 16), cot_batch('cd' * 16)
    start, end = original['memory_positions'][0][0]
    boundary = original['sample_lens'][0][0]
    assert original['labels'][0, start - 1:end + 1].eq(-100).all()
    assert original['latent_counts'] == [[2]]
    calls = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: calls.append(copy.deepcopy(args[0])))
    try:
        with torch.no_grad():
            before = model(**original).logits
            after = model(**changed).logits
        assert calls == [original['memory_token_ids'][0], changed['memory_token_ids'][0]]
        # Even the START position cannot see the later CoT embeddings.
        torch.testing.assert_close(before[:, :start], after[:, :start], rtol=0, atol=0)
        torch.testing.assert_close(before[:, boundary:15], after[:, boundary:15], rtol=0, atol=0)
        assert (before[:, end:boundary] - after[:, end:boundary]).abs().max() > 0

        # Loss on text before the CoT must not train the future-memory encoder.
        prefix = copy.deepcopy(original)
        prefix['labels'].fill_(-100)
        prefix['labels'][0, 2:4] = original['labels'][0, 2:4]
        model.zero_grad(set_to_none=True)
        model(**prefix).loss.backward()
        for module in (model.encoder, model.adapter):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.count_nonzero(g) == 0 for g in grads)

        # Loss on the continuation must train both encoder and adapter.
        continuation = copy.deepcopy(original)
        continuation['labels'][:, :end + 1] = -100
        continuation['labels'][:, boundary:] = -100
        model.zero_grad(set_to_none=True)
        loss = model(**continuation).loss
        assert torch.isfinite(loss)
        loss.backward()
        for module in (model.encoder, model.adapter):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(torch.count_nonzero(g) > 0 for g in grads)

        # Fused CE must apply the same shifted labels and preserve that gradient path.
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            reference = model(**continuation).loss
        model._liger_flce = LigerFusedLinearCrossEntropyLoss()
        fused = model(**continuation).loss
        torch.testing.assert_close(fused.float(), reference.float(), rtol=0.003, atol=0.003)
        fused.backward()
        for module in (model.encoder, model.adapter):
            assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in module.parameters())
    finally:
        hook.remove()
        clear_packed_sample_lens()

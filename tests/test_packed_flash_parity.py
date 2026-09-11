"""Check packed attention against independent document forwards and gradients."""
import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU FlashAttention check')
def test_packed_flash_matches_independent_documents_and_blocks_leakage():
    from latent_context.packed_flash import replace_with_packed_attention, set_packed_sample_lens, clear_packed_sample_lens
    torch.manual_seed(19)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        attention_dropout=0.0)
    config._attn_implementation = 'sdpa'
    reference = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).train()
    packed = copy.deepcopy(reference)
    replace_with_packed_attention(packed)
    ids = torch.randint(1, 128, (1, 24), device='cuda')
    labels = ids.clone()
    labels[:, [0, 4, 9, 17, 21, 22, 23]] = -100
    positions = torch.tensor([list(range(9)) + list(range(12)) + [0, 0, 0]], device='cuda')
    try:
        set_packed_sample_lens([[9, 12]])
        output = packed(input_ids=ids, labels=labels, position_ids=positions)
        refs = [reference(input_ids=ids[:, a:b], labels=labels[:, a:b]) for a, b in [(0, 9), (9, 21)]]
        expected_logits = torch.cat([out.logits for out in refs], dim=1)
        torch.testing.assert_close(output.logits[:, :21], expected_logits, rtol=0.04, atol=0.005)
        counts = [int((labels[:, a + 1:b] != -100).sum()) for a, b in [(0, 9), (9, 21)]]
        expected_loss = sum(out.loss * count for out, count in zip(refs, counts)) / sum(counts)
        torch.testing.assert_close(output.loss, expected_loss, rtol=0.003, atol=0.003)
        output.loss.backward()
        expected_loss.backward()
        for (name, left), (_, right) in zip(packed.named_parameters(), reference.named_parameters()):
            assert left.grad is not None and right.grad is not None, name
            torch.testing.assert_close(left.grad, right.grad, rtol=0.15, atol=0.002, msg=name)
        changed = ids.clone(); changed[:, :9] = 127 - changed[:, :9]
        with torch.no_grad():
            changed_logits = packed(input_ids=changed, position_ids=positions).logits
        torch.testing.assert_close(changed_logits[:, 9:21], output.logits[:, 9:21], rtol=0, atol=0)
    finally:
        clear_packed_sample_lens()

"""Decoder backends for online compaction.

Both backends turn a controller request into decoder input embeddings: the
prompt tokens, one ``<|memory_start|> <|memory|>*k <|memory_end|>`` span per
latent block (placeholders replaced by encoder latents) and the recent raw
tokens. Latents are encoded once per block and cached for the trajectory.

- ``VLLMBackend``: LCLM encoder in HF plus a vLLM decoder fed with
  ``prompt_embeds`` in the same process. Supports many concurrent
  trajectories (``submit_many``/``poll``); used for the paper's MATH runs.
- ``HFBackend``: single-trajectory reference on a loaded LCLM via
  ``decoder.generate(inputs_embeds=...)``. Simple, but slow for long outputs.
"""
import math
import os
import time

from .controller import MAX_CONTEXT, fingerprint

MEMORY_START = '<|memory_start|>'
MEMORY_END = '<|memory_end|>'
MEMORY_PLACEHOLDER = '<|memory|>'
# Sampling used for the paper's MATH compaction results.
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, seed=0)


def resolve_checkpoint(checkpoint):
    """Local checkpoint directory, downloading an HF repo id if needed."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=checkpoint, repo_type='model')


class LatentPromptBuilder:
    """Encode memory blocks and splice their latents into decoder input embeddings."""

    def __init__(self, model, tokenizer, embeddings, max_context=MAX_CONTEXT, sampling=None):
        import torch
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        self.max_context = max_context
        self.sampling = dict(SAMPLING if sampling is None else sampling)
        self.caches = {}
        for token in (MEMORY_START, MEMORY_END, MEMORY_PLACEHOLDER):
            assert len(tokenizer.encode(token, add_special_tokens=False)) == 1, token
        self.start_id = tokenizer.convert_tokens_to_ids(MEMORY_START)
        self.end_id = tokenizer.convert_tokens_to_ids(MEMORY_END)
        self.placeholder_id = tokenizer.convert_tokens_to_ids(MEMORY_PLACEHOLDER)

    def prefix(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids, final=False):
        return self.tokenizer.decode(ids, skip_special_tokens=final, clean_up_tokenization_spaces=False)

    def clear_cache(self, namespace=None):
        if namespace is None:
            self.caches.clear()
        else:
            self.caches.pop(namespace, None)

    def build(self, items):
        """``items``: list of ``(namespace, request)``. Returns ``[(embeds, metadata), ...]``.

        All latent blocks not yet cached for their namespace are encoded in one
        batched encoder call.
        """
        torch = self.torch
        ratio = self.model.encoder.compression_ratio
        embed_tokenizer = self.model.encoder.embed_tokenizer
        missing = {}
        for namespace, request in items:
            cache = self.caches.setdefault(namespace, {})
            for memory in request['memories']:
                assert self.decode(memory['token_ids']) == memory['text']
                key = fingerprint(memory['text'])
                if key not in cache and (namespace, key) not in missing:
                    enc = embed_tokenizer.encode(memory['text'], add_special_tokens=False)
                    assert enc
                    missing[(namespace, key)] = enc
        encoding_seconds = 0.0
        if missing:
            tic = time.monotonic()
            softs = self.model._process_latent_embeddings(list(missing.values()))
            assert len(softs) == len(missing)
            for ((namespace, key), enc), soft in zip(missing.items(), softs):
                assert soft.shape[0] == max(1, math.ceil(len(enc) / ratio)) and torch.isfinite(soft).all()
                self.caches[namespace][key] = (soft.detach().cpu(), len(enc))
            encoding_seconds = time.monotonic() - tic
        device = self.embeddings.weight.device
        results = []
        for namespace, request in items:
            ids = list(request['prefix_ids'])
            spans, blocks = [], []
            for memory in request['memories']:
                soft, encoder_tokens = self.caches[namespace][fingerprint(memory['text'])]
                ids.append(self.start_id)
                spans.append((len(ids), soft))
                ids.extend([self.placeholder_id] * len(soft))
                ids.append(self.end_id)
                blocks.append(dict(encoder_tokens=encoder_tokens, latent_tokens=len(soft)))
            ids.extend(request['recent_ids'])
            embeds = self.embeddings(torch.tensor(ids, device=device))
            for start, soft in spans:
                embeds[start:start + len(soft)] = soft.to(device=embeds.device, dtype=embeds.dtype)
            assert torch.isfinite(embeds).all()
            results.append((embeds, dict(prompt_positions=len(ids), memory_blocks=blocks,
                                         encoding_seconds=encoding_seconds)))
        return results


class HFBackend(LatentPromptBuilder):
    """Serial reference backend on a loaded LCLM (``latent_context.from_pretrained``)."""

    def __init__(self, model, tokenizer, max_context=MAX_CONTEXT, sampling=None):
        super().__init__(model, tokenizer, model.decoder.get_input_embeddings(), max_context, sampling)
        eos = model.decoder.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_ids[0]

    def generate(self, request, max_tokens):
        torch = self.torch
        s = self.sampling
        with torch.inference_mode():
            (embeds, metadata), = self.build([('serial', request)])
            assert metadata['prompt_positions'] + max_tokens <= self.max_context, 'No silent context truncation'
            if s.get('seed') is not None:
                torch.manual_seed(s['seed'])
            kwargs = dict(do_sample=False)
            if s['temperature'] > 0:
                kwargs = dict(do_sample=True, temperature=s['temperature'], top_p=s['top_p'],
                              top_k=s['top_k'], min_p=s['min_p'])
            out = self.model.decoder.generate(
                inputs_embeds=embeds[None], attention_mask=torch.ones((1, len(embeds)), dtype=torch.long, device=embeds.device),
                max_new_tokens=max_tokens, repetition_penalty=1.0, eos_token_id=self.eos_ids, pad_token_id=self.pad_id, **kwargs)
        tokens = out[0].tolist()
        return dict(metadata, token_ids=tokens, finish_reason='stop' if tokens[-1] in self.eos_ids else 'length')


class VLLMBackend(LatentPromptBuilder):
    """HF encoder + vLLM decoder on one GPU, serving many trajectories at once."""

    def __init__(self, checkpoint, max_context=MAX_CONTEXT, max_active=16, sampling=None,
                 gpu_memory_utilization=0.70, max_num_batched_tokens=8192, max_encode_batch_size=32):
        import json
        import gc
        import torch
        from vllm import LLM
        from inference.hf import load_model
        checkpoint = resolve_checkpoint(checkpoint)
        config = json.loads(open(os.path.join(checkpoint, 'decoder', 'config.json')).read())
        assert config['max_position_embeddings'] >= max_context
        model, tokenizer, _ = load_model(checkpoint, device='cuda:0', dtype='bf16')
        model.encoder.max_encode_batch_size = max_encode_batch_size
        embeddings = model.decoder.get_input_embeddings()
        # vLLM serves the decoder; keep only its input embeddings in HF.
        model.decoder = None
        gc.collect()
        torch.cuda.empty_cache()
        super().__init__(model, tokenizer, embeddings, max_context, sampling)
        self.max_active = max_active
        self.llm = LLM(model=os.path.join(checkpoint, 'decoder'), dtype='bfloat16', tensor_parallel_size=1,
                       enable_prompt_embeds=True, enable_prefix_caching=False,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_context,
                       max_num_seqs=max_active, max_num_batched_tokens=max_num_batched_tokens,
                       enable_chunked_prefill=True, seed=0)
        self.engine = self.llm.llm_engine
        self.pending = {}
        self.serial_counter = 0

    def clear_cache(self, namespace=None):
        assert all(p['namespace'] != namespace for p in self.pending.values()) if namespace else not self.pending
        super().clear_cache(namespace)

    def _params(self, max_tokens):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        s = self.sampling
        return SamplingParams(max_tokens=max_tokens, temperature=s['temperature'], top_p=s['top_p'], top_k=s['top_k'],
                              min_p=s['min_p'], presence_penalty=0.0, repetition_penalty=1.0, seed=s.get('seed'),
                              skip_special_tokens=False, output_kind=RequestOutputKind.FINAL_ONLY)

    def submit_many(self, items):
        """``items``: dicts with ``namespace``, ``request_id``, ``request`` and ``allowance``."""
        assert len(self.pending) + len(items) <= self.max_active
        namespaces = {x['namespace'] for x in items}
        assert len(namespaces) == len(items) and not namespaces & {p['namespace'] for p in self.pending.values()}
        start = time.monotonic()
        with self.torch.inference_mode():
            built = self.build([(x['namespace'], x['request']) for x in items])
            for item, (embeds, metadata) in zip(items, built):
                assert metadata['prompt_positions'] + item['allowance'] <= self.max_context, 'No silent context truncation'
                rid = item['request_id']
                assert rid not in self.pending
                self.pending[rid] = dict(namespace=item['namespace'], metadata=metadata, start=start)
                self.engine.add_request(rid, {'prompt_embeds': embeds.detach().float().cpu()}, self._params(item['allowance']))

    def poll(self):
        """Advance the engine one step; return ``[(request_id, output), ...]`` for finished requests."""
        assert self.pending
        completed = []
        for response in self.engine.step():
            if not response.finished:
                continue
            assert response.request_id in self.pending and len(response.outputs) == 1
            record = self.pending.pop(response.request_id)
            out = response.outputs[0]
            completed.append((response.request_id, dict(record['metadata'], token_ids=list(out.token_ids),
                                                        finish_reason=out.finish_reason,
                                                        elapsed_seconds=time.monotonic() - record['start'])))
        return completed

    def generate(self, request, max_tokens):
        assert not self.pending
        self.serial_counter += 1
        rid = f'serial-{self.serial_counter}'
        self.submit_many([dict(namespace='serial', request_id=rid, request=request, allowance=max_tokens)])
        while True:
            finished = self.poll()
            if finished:
                assert len(finished) == 1 and finished[0][0] == rid
                return finished[0][1]

"""LCLM generation backend for multi-turn tool episodes, and a one-replica-per-GPU job pool.

Each memory span in the chat prompt is encoded by the LCLM encoder/adapter into
``ceil(tokens / compression_ratio)`` latents, spliced into the decoder input
embeddings, and decoded by vLLM from prompt embeddings.
"""
from __future__ import annotations

import gc
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue as queue_lib
import time
import traceback


def resolve_checkpoint(checkpoint):
    """Local checkpoint directory, downloading a Hugging Face repo id if needed."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=checkpoint, repo_type='model')


class Backend:
    def __init__(self, checkpoint, max_tokens=512, max_model_len=65536,
                 gpu_memory_utilization=0.65, seed=0):
        import torch
        from inference.hf import load_model
        from vllm import LLM, SamplingParams
        self.torch = torch
        checkpoint = resolve_checkpoint(checkpoint)
        self.model, self.tokenizer, _ = load_model(checkpoint, device='cuda:0', dtype='bf16')
        self.compression_ratio = self.model.encoder.compression_ratio
        self.model.encoder.max_encode_batch_size = 32
        self.embeddings = self.model.decoder.get_input_embeddings()
        # vLLM serves the decoder; keep only the encoder, adapter and input embeddings here.
        self.model.decoder = None
        gc.collect()
        torch.cuda.empty_cache()
        self.llm = LLM(model=str(Path(checkpoint)/'decoder'), dtype='bfloat16',
                       tensor_parallel_size=1, enable_prompt_embeds=True,
                       enable_prefix_caching=False, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, max_num_seqs=8, enforce_eager=False, seed=seed)
        self.params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0,
                                     repetition_penalty=1.0, seed=seed)
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.cache = {}
        for token in ('<|memory_start|>', '<|memory_end|>', '<|memory|>'):
            assert len(self.tokenizer.encode(token, add_special_tokens=False)) == 1

    def generate(self, messages, tools):
        from inference.vllm_inference._prompt_embeds import _expand_memory_placeholders, _find_placeholder_spans
        torch = self.torch
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools or None,
                                                    tokenize=False, add_generation_prompt=True)
        assert '<think>' not in prompt and '</think>' not in prompt
        expanded, contents = _expand_memory_placeholders(prompt, self.model.encoder.embed_tokenizer,
                                                         self.compression_ratio)
        ids = self.tokenizer.encode(expanded, add_special_tokens=True)
        spans = _find_placeholder_spans(ids, self.tokenizer)
        assert len(spans) == len(contents)
        if len(ids) + self.max_tokens > self.max_model_len:
            raise ValueError(f'Context would overflow: {len(ids)} + {self.max_tokens} > {self.max_model_len}')
        started = time.monotonic()
        with torch.inference_mode():
            embeds = self.embeddings(torch.tensor(ids, device='cuda:0'))
            for (start, end), content in zip(spans, contents):
                digest = hashlib.sha256(content.encode()).hexdigest()
                if digest not in self.cache:
                    token_ids = self.model.encoder.embed_tokenizer.encode(content, add_special_tokens=False)
                    soft = self.model._process_latent_embeddings([token_ids])[0]
                    assert soft.shape[0] == max(1, math.ceil(len(token_ids)/self.compression_ratio))
                    assert torch.isfinite(soft).all()
                    self.cache[digest] = soft.detach().cpu()
                soft = self.cache[digest]
                assert end - start == soft.shape[0], 'Memory spans must align exactly'
                embeds[start:end] = soft.to(embeds.device, dtype=embeds.dtype)
            assert torch.isfinite(embeds).all()
            cpu_embeds = embeds.detach().float().cpu()
        output = self.llm.generate([{'prompt_embeds': cpu_embeds}], self.params, use_tqdm=False)[0].outputs[0]
        return dict(text=output.text, finish_reason=output.finish_reason, output_tokens=len(output.token_ids),
                    prompt_positions=len(ids), uncompressed_prompt_tokens=len(self.tokenizer.encode(prompt)),
                    memory_spans=len(spans), elapsed_seconds=time.monotonic()-started)


def _worker(gpu, jobs, results, gate, backend_kwargs, job_fn):
    # GPU visibility must be fixed before torch or vLLM is imported in this process.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['OMP_NUM_THREADS'] = '1'
    try:
        with gate:
            backend = Backend(**backend_kwargs)
        while True:
            job = jobs.get()
            if job is None:
                break
            backend.cache.clear()
            results.put(('rows', job_fn(job, backend)))
        results.put(('done', gpu))
    except BaseException:
        results.put(('error', traceback.format_exc()))
        raise


def run_jobs(jobs, job_fn, backend_kwargs, gpus, on_result):
    """Run ``job_fn(job, backend)`` for every job and pass each return value to ``on_result``.

    ``gpus == 1`` runs in this process. Otherwise one independent replica runs per
    GPU (at most two load at once); ``job_fn`` must be a module-level function.
    """
    if not jobs:
        return
    if gpus == 1:
        backend = Backend(**backend_kwargs)
        for job in jobs:
            backend.cache.clear()
            on_result(job_fn(job, backend))
        return
    ctx = mp.get_context('spawn')
    job_queue, results = ctx.Queue(), ctx.Queue()
    gate = ctx.BoundedSemaphore(2)
    for job in jobs:
        job_queue.put(job)
    for _ in range(gpus):
        job_queue.put(None)
    workers = [ctx.Process(target=_worker, args=(gpu, job_queue, results, gate, backend_kwargs, job_fn))
               for gpu in range(gpus)]
    for process in workers:
        process.start()
    remaining, finished = len(jobs), 0
    try:
        while remaining or finished < gpus:
            try:
                kind, value = results.get(timeout=30)
            except queue_lib.Empty:
                if any(p.exitcode not in (None, 0) for p in workers):
                    raise RuntimeError('A GPU worker exited unexpectedly')
                continue
            if kind == 'error':
                raise RuntimeError('GPU worker failed:\n' + value)
            if kind == 'rows':
                remaining -= 1
                on_result(value)
            else:
                finished += 1
    finally:
        # vLLM engine children can keep a finished worker alive; bound the shutdown.
        deadline = time.monotonic() + 30
        for process in workers:
            process.join(timeout=max(0.1, deadline - time.monotonic()))
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

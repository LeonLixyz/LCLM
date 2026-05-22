"""Mirror the four LCLM eval datasets to latent-context/lclm-eval.

One HF dataset, four sub-configs: ruler / gsm8k / longhealth5 / longbench.
All share the same {prompt, category, extra_info} schema.

Source mapping:
    ruler        ← tonychenxyz/ruler-full          @ memwrap, split=validation
    gsm8k        ← tonychenxyz/codellava-gsm8k-memwrap (single split)
    longhealth5  ← leonli66/longhealth5            @ memwrap, split=test
    longbench    ← nimitkalra/LongBench-v1         @ memwrap, split=validation

Run:
    modal run --detach _modal_run/lclm_mirror_eval_datasets.py
"""
from __future__ import annotations
import modal

app = modal.App("lclm-mirror-eval-datasets")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets>=2.18", "huggingface-hub>=0.25", "pyarrow")
)

TARGET = "latent-context/lclm-eval"

SOURCES = {
    "ruler":       {"repo": "tonychenxyz/ruler-full",          "config": "memwrap", "split": "validation"},
    "gsm8k":       {"repo": "tonychenxyz/codellava-gsm8k-memwrap", "config": None,  "split": "train"},
    "longhealth5": {"repo": "leonli66/longhealth5",            "config": "memwrap", "split": "test"},
    "longbench":   {"repo": "nimitkalra/LongBench-v1",         "config": "memwrap", "split": "validation"},
}

README = """---
license: other
language:
  - en
tags:
  - long-context
  - evaluation
  - lclm
configs:
  - config_name: ruler
    data_files:
      - split: test
        path: ruler/test-*.parquet
  - config_name: gsm8k
    data_files:
      - split: test
        path: gsm8k/test-*.parquet
  - config_name: longhealth5
    data_files:
      - split: test
        path: longhealth5/test-*.parquet
  - config_name: longbench
    data_files:
      - split: test
        path: longbench/test-*.parquet
---

# LCLM evaluation datasets

Unified eval mix for **Latent Context Language Models (LCLM)**. Four
benchmarks, one schema (`{prompt, category, extra_info}`), one repo.

| Config | Source | Rows | Notes |
|---|---|---:|---|
| `ruler` | `tonychenxyz/ruler-full` (memwrap, validation) | 39,000 | 13 tasks × 6 ctx lengths × 500 |
| `gsm8k` | `tonychenxyz/codellava-gsm8k-memwrap` | 1,319 | grade-school math word problems |
| `longhealth5` | `leonli66/longhealth5` (memwrap, test) | 400 | 5-doc patient-record QA |
| `longbench` | `nimitkalra/LongBench-v1` (memwrap, validation) | 4,750 | 21 English+Chinese long-context tasks |

## Schema

Every row has three columns:

- `prompt` (`str`): full chat-formatted prompt with
  `<|memory_start|>...<|memory_end|>` markers wrapping the context to
  be compressed by the LCLM encoder.
- `category` (`str`): task-and-length tag (e.g. `niah_single_1_4096`,
  `narrativeqa`).
- `extra_info` (`dict`): per-task metadata including
  `ground_truth.answers` (list of acceptable strings),
  `scoring_function` (string-match flavor), and original-task fields.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("latent-context/lclm-eval", "ruler", split="test")
print(ds[0]["prompt"][:200])
print(ds[0]["category"])
print(ds[0]["extra_info"]["ground_truth"])
```

## Scoring

The LCLM eval pipeline reads `extra_info.scoring_function` per sample.
For RULER subtasks this is `ruler_string_match_all` /
`ruler_string_match_part` (official NVIDIA RULER reference impl,
case-insensitive substring match). For other benchmarks see the LCLM
benchmark code.

## Companion code

- Inference + eval: <https://github.com/LeonLixyz/LCLM>
- Checkpoints: `latent-context/0.6b-4b-LCLM-{4,8,16}x`
"""


@app.function(image=image, timeout=2*60*60,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def mirror():
    import os, tempfile, shutil
    from datasets import load_dataset, DatasetDict
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(TARGET, repo_type="dataset", exist_ok=True,
                token=os.environ["HF_TOKEN"])

    tmp = tempfile.mkdtemp(prefix="lclm-eval-mirror-")
    try:
        for cfg_name, src in SOURCES.items():
            print(f"\n>>> {cfg_name} ← {src['repo']}", flush=True)
            ds = load_dataset(
                src["repo"],
                src["config"],
                split=src.get("split"),
            )
            print(f"    rows={len(ds)}, cols={ds.column_names}", flush=True)
            # Normalize to a single 'test' split, write as parquet under <config>/
            out_dir = os.path.join(tmp, cfg_name)
            os.makedirs(out_dir, exist_ok=True)
            parquet_path = os.path.join(out_dir, "test-00000-of-00001.parquet")
            ds.to_parquet(parquet_path)
            print(f"    wrote {parquet_path}", flush=True)

        # README
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(README)

        print(f"\n>>> Uploading to {TARGET}", flush=True)
        api.upload_folder(
            repo_id=TARGET,
            repo_type="dataset",
            folder_path=tmp,
            token=os.environ["HF_TOKEN"],
            commit_message="mirror eval datasets to latent-context",
            ignore_patterns=["state.json"],
        )
        print("✓ done", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
